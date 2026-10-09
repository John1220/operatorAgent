"""Real in-memory transaction, JSON/view decoder and resident pipe acceptance.

CPU protocol coverage is not native CUDA or performance evidence.
"""
import base64
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import zlib
from unittest.mock import Mock

import numpy as np
import pytest

from operator_agent import session_inputs as si
from operator_agent.input_fidelity import describe_inputs
from operator_agent.remote_input_fidelity import decode_json_inputs, JSON_PROTOCOL
from operator_agent.remote_executor import _RemoteSession, SSHExecutor, SessionError
from operator_agent.measurement import TRANSPORT_UNKNOWN
from operator_agent.run_contract import protocol_identity


def prepared(inputs):
    metadata, raw = si.json_input_parts(inputs)
    return si.PreparedInputs(metadata, raw)


def upload(cache, packet):
    begin = cache.begin(cache.session_id, packet)
    raw = {r["sha256"]: b for r, b in zip(packet["storages"], packet.raw_storages)}
    for ref in begin["missing"]:
        value = raw[ref["sha256"]]
        for offset in range(0, len(value), si.CHUNK_BYTES):
            cache.chunk(cache.session_id, packet["hash"], ref["sha256"], offset,
                        base64.b64encode(value[offset:offset+si.CHUNK_BYTES]).decode())
    cache.commit(cache.session_id, packet["hash"])
    return begin


def resolved(cache, packet):
    wire, proof = cache.resolve(dict(packet, session_id=cache.session_id))
    assert proof["content_verified"] is True
    return decode_json_inputs(wire)


@pytest.mark.parametrize("layout", ["contiguous", "transpose", "offset", "strided", "broadcast", "negative", "readonly"])
def test_cached_decode_preserves_actual_storage_layout_alias_and_scalars(layout):
    root = np.arange(24, dtype=np.float32).reshape(4, 6)
    x = root
    if layout == "transpose": x = root.T
    if layout == "offset": x = root[1:, 1:]
    if layout == "strided": x = root[:, ::2]
    if layout == "broadcast": x = np.broadcast_to(root[0], (3, 6))
    if layout == "negative": x = root[::-1, ::-1]
    if layout == "readonly": x.setflags(write=False)
    inputs = {"X": x, "alias": x, "epsilon": -0.0, "nan": np.float64("nan"),
              "complex": 1+2j, "bytes": b"\x00\xff", "label": "输入", "bool": True}
    cache, packet = si.SessionInputs(), prepared(inputs)
    upload(cache, packet)
    actual = resolved(cache, packet)
    assert describe_inputs(actual) == describe_inputs(inputs)
    assert np.shares_memory(actual["X"], actual["alias"])
    assert len(json.dumps(packet)) < 3000  # No raw data/base64 storage in descriptor.
    assert not any(type(v) is bytes for v in packet.values())


def test_equal_content_dedup_does_not_invent_aliasing_and_candidate_cannot_mutate_cache():
    inputs = {"A": np.arange(16, dtype=np.float32), "B": np.arange(16, dtype=np.float32)}
    packet, cache = prepared(inputs), si.SessionInputs()
    assert len(upload(cache, packet)["missing"]) == 1
    first = resolved(cache, packet)
    assert not np.shares_memory(first["A"], first["B"])
    first["A"][:] = 123
    assert describe_inputs(resolved(cache, packet)) == describe_inputs(inputs)
    assert upload(cache, packet)["missing"] == []


def test_small_inputs_keep_inline_and_large_inputs_are_json_safe_descriptors():
    assert si.prepare_inputs({"X": np.arange(4, dtype=np.float32)})["protocol"] == JSON_PROTOCOL
    x = np.arange(si.CACHE_THRESHOLD, dtype=np.uint8)
    packet = si.prepare_inputs({"X": x})
    assert isinstance(packet, si.PreparedInputs) and packet["protocol"] == si.PROTOCOL
    assert len(json.dumps(packet)) < 500
    assert packet["storages"][0]["sha256"] == hashlib.sha256(x.tobytes()).hexdigest()


@pytest.mark.parametrize("change", ["hash", "protocol", "size-bool", "size-negative", "size-large", "extra", "scalars"])
def test_bad_declarations_cannot_reserve_or_promote_inputs(change):
    packet, cache = prepared({"X": np.arange(4, dtype=np.float32)}), si.SessionInputs()
    data = copy.deepcopy(dict(packet))
    if change == "hash": data["hash"] = "0"*64
    elif change == "protocol": data["protocol"] = "other"
    elif change == "size-bool": data["storages"][0]["bytes"] = True
    elif change == "size-negative": data["storages"][0]["bytes"] = -1
    elif change == "size-large": data["storages"][0]["bytes"] = si.MAX_STORAGE_BYTES + 1
    elif change == "extra": data["path"] = "/untrusted"
    else: data["scalars"] = []
    if change != "hash": data["hash"] = si.digest({k:v for k,v in data.items() if k != "hash"})
    with pytest.raises(si.SessionInputError): cache.begin(cache.session_id, data)
    assert not cache.blobs and cache.ready is None


@pytest.mark.parametrize("change", ["incomplete", "wrong-digest", "bad-base64", "overlap", "offset-bool", "oversize"])
def test_partial_corrupt_or_replayed_upload_never_becomes_candidate_input(change):
    cache, packet = si.SessionInputs(), prepared({"X": np.arange(4, dtype=np.float32)})
    cache.begin(cache.session_id, packet)
    sha = packet["storages"][0]["sha256"]
    raw = packet.raw_storages[0]
    if change in {"incomplete", "wrong-digest"}:
        sent = raw[:4] if change == "incomplete" else b"x" * len(raw)
        cache.chunk(cache.session_id, packet["hash"], sha, 0, base64.b64encode(sent).decode())
        with pytest.raises(si.SessionInputError): cache.commit(cache.session_id, packet["hash"])
    else:
        offset = True if change == "offset-bool" else 1 if change == "overlap" else 0
        data = "!" if change == "bad-base64" else base64.b64encode(raw + b"extra" if change == "oversize" else raw).decode()
        with pytest.raises(si.SessionInputError): cache.chunk(cache.session_id, packet["hash"], sha, offset, data)
    assert not cache.blobs
    with pytest.raises(si.SessionInputError): resolved(cache, packet)
    cache.abort()
    assert cache.pending is None


def test_cached_corruption_and_stale_session_refs_hold_instead_of_silent_repair():
    cache, packet = si.SessionInputs(), prepared({"X": np.arange(4, dtype=np.float32)})
    upload(cache, packet)
    cache.blobs[packet["storages"][0]["sha256"]] = b"x" * 16
    with pytest.raises(si.SessionInputError): resolved(cache, packet)
    with pytest.raises(si.SessionInputError): upload(cache, packet)
    other = si.SessionInputs()
    upload(other, packet)
    with pytest.raises(si.SessionInputError): other.resolve(dict(packet, session_id=cache.session_id))
    cache.close()
    assert not cache.blobs and cache.ready is None and cache.pending is None
    with pytest.raises(si.SessionInputError): cache.begin(cache.session_id, packet)


def test_lru_is_bounded_and_current_batch_cannot_evict_its_own_dependencies():
    cache = si.SessionInputs(capacity=64)
    a, b, c = [prepared({"X": np.arange(8, dtype=np.float32)+v}) for v in (0, 10, 20)]
    upload(cache, a); upload(cache, b); upload(cache, a); upload(cache, c)
    assert a["storages"][0]["sha256"] in cache.blobs
    assert b["storages"][0]["sha256"] not in cache.blobs
    assert sum(map(len, cache.blobs.values())) == 64
    too_large = prepared({"X": np.arange(32, dtype=np.float32)})
    with pytest.raises(si.SessionInputError): cache.begin(cache.session_id, too_large)


def test_metadata_validation_precedes_atomic_promotion():
    cache, packet = si.SessionInputs(), prepared({"X": np.arange(4, dtype=np.float32)})
    packet["tensors"]["X"]["offset"] = 1000
    packet["hash"] = si.digest({k:v for k,v in packet.items() if k != "hash"})
    with pytest.raises(ValueError): upload(cache, packet)
    assert not cache.blobs and cache.ready is None


def test_new_metadata_batch_must_commit_even_if_all_bytes_are_cached():
    cache, packet = si.SessionInputs(), prepared({"X": np.arange(4,dtype=np.float32)})
    upload(cache, packet)
    changed = prepared({"X": np.arange(4,dtype=np.float32), "scale": 2})
    assert cache.begin(cache.session_id, changed)["missing"] == []
    with pytest.raises(si.SessionInputError): resolved(cache, changed)
    cache.commit(cache.session_id, changed["hash"])
    assert resolved(cache, changed)["scale"] == 2


@pytest.mark.parametrize("offset", [0, True])
def test_replayed_or_boolean_chunk_offset_is_rejected_before_commit(offset):
    cache, packet = si.SessionInputs(), prepared({"X": np.arange(4,dtype=np.float32)})
    cache.begin(cache.session_id, packet)
    sha = packet["storages"][0]["sha256"]
    data = base64.b64encode(packet.raw_storages[0][:4]).decode()
    cache.chunk(cache.session_id, packet["hash"], sha, 0, data)
    with pytest.raises(si.SessionInputError): cache.chunk(cache.session_id, packet["hash"], sha, offset, data)


@pytest.mark.parametrize("encoding", ["raw", "zlib"])
def test_chunk_encoding_is_lossless_and_hashes_uncompressed_bytes(encoding):
    inputs = {"X": np.tile(np.array([-1,0,1,2],np.float16),(32,32))}
    packet, cache = prepared(inputs), si.SessionInputs()
    cache.begin(cache.session_id, packet)
    raw = packet.raw_storages[0]
    data = zlib.compress(raw,1) if encoding == "zlib" else raw
    ack = cache.chunk(cache.session_id,packet["hash"],packet["storages"][0]["sha256"],0,
                      base64.b64encode(data).decode(),encoding)
    assert ack["next_offset"] == len(raw)
    cache.commit(cache.session_id,packet["hash"])
    assert describe_inputs(resolved(cache,packet)) == describe_inputs(inputs)


@pytest.mark.parametrize("variant", ["bomb", "truncated", "trailing", "concatenated", "invalid", "unknown-codec"])
def test_compressed_chunks_are_bounded_complete_single_streams(variant):
    packet, cache = prepared({"X": np.arange(4,dtype=np.float32)}), si.SessionInputs()
    cache.begin(cache.session_id,packet)
    data = zlib.compress(packet.raw_storages[0])
    encoding = "zlib"
    if variant == "bomb": data = zlib.compress(b"x"*(si.CHUNK_BYTES*2))
    elif variant == "truncated": data = data[:-1]
    elif variant == "trailing": data += b"tail"
    elif variant == "concatenated": data += data
    elif variant == "invalid": data = b"bad zlib"
    else: encoding = "arbitrary"
    with pytest.raises(si.SessionInputError):
        cache.chunk(cache.session_id,packet["hash"],packet["storages"][0]["sha256"],0,
                    base64.b64encode(data).decode(),encoding)
    assert not cache.blobs


def test_decompression_cannot_allocate_unbounded_output_or_flush(monkeypatch):
    original = si.zlib.decompressobj
    class BoundedDecoder:
        def __init__(self): self.inner = original()
        def decompress(self,data,max_length=0):
            assert max_length == si.CHUNK_BYTES+1, "unbounded decompression allocation"
            return self.inner.decompress(data,max_length)
        def flush(self): raise AssertionError("unbounded flush")
        def __getattr__(self,name): return getattr(self.inner,name)
    monkeypatch.setattr(si.zlib,"decompressobj",BoundedDecoder)
    packet, cache = prepared({"X": np.arange(4,dtype=np.float32)}), si.SessionInputs()
    cache.begin(cache.session_id,packet)
    data = base64.b64encode(zlib.compress(packet.raw_storages[0])).decode()
    cache.chunk(cache.session_id,packet["hash"],packet["storages"][0]["sha256"],0,data,"zlib")


@pytest.fixture
def real_session(tmp_path, monkeypatch):
    # Deploy exactly the shared production sources, no repo import fallback.
    import shutil
    for local, remote in SSHExecutor(host="offline")._remote_files():
        shutil.copy2(local, tmp_path / Path(remote).name)
    monkeypatch.setenv("OPERATOR_AGENT_GPU_CLAIM_DIR", str(tmp_path / "claims"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    s = _RemoteSession([sys.executable, "-u", str(tmp_path / "runner_server.py")], 30, "local-server")
    s.start()
    assert s.call({"action": "ping"})["input_cache"]["protocol"] == si.PROTOCOL
    yield s
    assert s.close(grace=3)["local_quiescent"]


GOOD = "import torch\ndef kernel(X): return X + 0\n"


def verify(s, packet, code=GOOD):
    return s.call({"action": "verify", "inputs": packet, "code": code,
                   "kernel_type": "custom", "reference_source": "def reference(X): return X",
                   "precision": "fp32", "max_error": 1e-5})


def test_real_deployed_server_uploads_once_reuses_and_still_reads_actual_torch_inputs(real_session):
    x = np.arange(512*1024, dtype=np.float32).reshape(1024,512)
    packet = prepared({"X": x})
    first = verify(real_session, packet)
    second = verify(real_session, packet)
    assert first["success"] and second["success"]
    assert first["input_transfer"]["uploaded_bytes"] == x.nbytes
    assert first["input_transfer"]["uploaded_chunks"] == 4
    assert second["input_transfer"]["uploaded_bytes"] == 0
    assert second["input_transfer"]["reused_storages"] == 1
    assert second["input_fidelity"]["observed"] == describe_inputs({"X": x})
    # Cache is not reused as a numerical verdict.
    bad = verify(real_session, packet, "import torch\ndef kernel(X): return X + 1\n")
    assert bad["success"] is False and bad["max_error"] == 1
    assert bad["input_transfer"]["uploaded_bytes"] == 0 and real_session.alive


@pytest.mark.parametrize("compressible", [True, False])
def test_real_server_uses_compression_only_when_smaller_and_keeps_actual_readback(real_session, compressible):
    x = (np.tile(np.array([-1,0,1,2],np.float16),(128,512)) if compressible else
         np.random.default_rng(42).integers(0,256,(16,64),dtype=np.uint8))
    result = verify(real_session,prepared({"X":x}))
    assert result["success"] and result["input_fidelity"]["observed"] == describe_inputs({"X":x})
    proof = result["input_transfer"]
    assert proof["uploaded_bytes"] == x.nbytes
    assert proof["uploaded_chunks"] == 1
    if compressible:
        assert proof["chunk_encodings"] == {"raw":0,"zlib":1}
        assert proof["uploaded_wire_bytes"] < x.nbytes/10
    else:
        assert proof["chunk_encodings"] == {"raw":1,"zlib":0}
        assert proof["uploaded_wire_bytes"] >= x.nbytes


def test_real_server_requires_commit_and_nonce_before_any_candidate_exec(real_session, tmp_path):
    packet = prepared({"X": np.arange(4, dtype=np.float32)})
    real_session.call({"action": "input_begin", "session_id": real_session.input_cache["session_id"], "manifest": dict(packet)})
    touched = tmp_path / "candidate-ran"
    code = f"from pathlib import Path\nPath({str(touched)!r}).touch()\ndef kernel(X): return X\n"
    r = verify(real_session, dict(packet, session_id=real_session.input_cache["session_id"]), code)
    assert r["input_identity_unknown"] is True and not touched.exists()
    r = verify(real_session, dict(packet, session_id="0"*32), code)
    assert r["input_identity_unknown"] is True and not touched.exists()


@pytest.mark.parametrize("phase", ["handshake", "codecs", "begin", "chunk", "commit", "final"])
def test_client_receipt_guard_stops_unknown_inputs_without_candidate_retry(monkeypatch, phase):
    s = _RemoteSession([], 1, "offline")
    s.state = "ready"
    session_id = "a"*32
    packet = prepared({"X": np.arange(4, dtype=np.float32)})
    s.input_cache = {"protocol": si.PROTOCOL, "session_id": session_id, "chunk_bytes": si.CHUNK_BYTES,
                     "chunk_encodings": ["raw", "zlib"]}
    calls = []
    def reply(payload, **kwargs):
        calls.append(payload["action"])
        result = {"success": True, "session_id": session_id, "manifest_hash": packet["hash"]}
        if payload["action"] == "input_begin": result["missing"] = packet["storages"]
        if payload["action"] == "input_chunk": result.update(storage_hash=packet["storages"][0]["sha256"],next_offset=16)
        if payload["action"] == "verify": result["input_transfer"] = dict(result, protocol=si.PROTOCOL, content_verified=True, storage_count=1, storage_bytes=16)
        bad_action = {"begin":"input_begin","chunk":"input_chunk","commit":"input_commit","final":"verify"}.get(phase)
        if payload["action"] == bad_action: result["manifest_hash"] = "0"*64; result.get("input_transfer", {}).update(manifest_hash="0"*64)
        return result
    monkeypatch.setattr(s, "_call_locked", reply)
    if phase == "handshake": s.input_cache = {}
    if phase == "codecs": del s.input_cache["chunk_encodings"]
    with pytest.raises(SessionError): s.call({"action":"verify","inputs":packet})
    assert s.state == "broken"
    assert calls.count("verify") == (1 if phase == "final" else 0)


def test_upload_commit_and_final_rpc_do_not_get_independent_deadlines():
    peer = '''import sys,json,time
from operator_agent.session_inputs import SessionInputs
c=SessionInputs()
for line in sys.stdin:
 r=json.loads(line); a=r['action']
 if a=='ping': out=dict(pong=True,input_cache=c.advertisement())
 elif a=='input_begin': out=c.begin(r['session_id'],r['manifest'])
 elif a=='input_chunk':
  time.sleep(.08); out=c.chunk(r['session_id'],r['manifest_hash'],r['storage_hash'],r['offset'],r['data'],r['encoding'])
 elif a=='input_commit':
  time.sleep(.4); out=c.commit(r['session_id'],r['manifest_hash'])
 elif a=='shutdown': out=dict(shutdown=True)
 else: raise AssertionError('candidate ran after upload deadline')
 print(json.dumps(dict(id=r['id'],result=out)),flush=True)
 if a=='shutdown': break
'''
    s = _RemoteSession([sys.executable,"-u","-c",peer], 3, "slow-upload")
    s.start()
    try:
        s.call({"action":"ping"})
        packet = prepared({"X": np.arange(3*si.CHUNK_BYTES,dtype=np.uint8)})
        started = time.monotonic()
        with pytest.raises(SessionError) as exc:
            s.call({"action":"verify","inputs":packet},timeout=.3)
        assert .25 < time.monotonic()-started < .55
        assert exc.value.session_phase == "response" and s.state == "broken"
        assert s._next_id <= 6  # ping + begin + three chunks + commit; no candidate RPC.
    finally:
        assert s.close(grace=.3)["local_quiescent"]


def test_cache_failures_are_unknown_to_production_verifier_not_candidate_failures(monkeypatch):
    from operator_agent import IntentParser, Verifier
    from operator_agent.failure_route import classify, ROUTE_HOLD
    from operator_agent.schemas import VerificationLevel
    backend = SSHExecutor(host="offline", persistent=True)
    backend._deployed = True
    backend.compile = Mock(return_value={"success": True})
    monkeypatch.setattr(backend, "_session_call", Mock(side_effect=SessionError("broken","digest mismatch",phase="input_cache")))
    t = IntentParser().parse("relu fp32 python nvidia")
    results = Verifier(backend=backend).verify(GOOD, t, {"X": np.arange(4,dtype=np.float32).reshape(2,2)})
    assert results[1].level == VerificationLevel.L2_FUNCTION
    assert results[1].metrics["correctness"]["verdict"] == "unknown"
    assert results[1].metrics["transport_outcome"] == TRANSPORT_UNKNOWN
    assert classify(results).route == ROUTE_HOLD


def test_new_session_sources_are_deployed_and_frozen_protocol_bound():
    files = [Path(remote).name for _, remote in SSHExecutor(host="offline")._remote_files()]
    assert files.index("session_inputs.py") < files.index("runner_server.py")
    assert "operator_agent/session_inputs.py" in protocol_identity()
