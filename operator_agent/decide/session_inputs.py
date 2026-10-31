"""Bounded, content-addressed inputs owned by one resident runner's memory.

Only transport is cached, never tensors or candidate outputs. Each use verifies
complete immutable bytes and passes the original scalar/storage/view decoder.
Not a hostile same-UID/Python sandbox or a durable/cross-session blob store.
"""
from __future__ import annotations

import base64
from collections import OrderedDict
import hashlib
import json
import secrets
import zlib

try:
    from ..decide.input_fidelity import MAX_STORAGE_BYTES, InputFidelityError
    from ..decide.remote_input_fidelity import (json_input_parts, ResolvedJSONInputs, decode_json_inputs)
except ImportError:  # deployed siblings
    from input_fidelity import MAX_STORAGE_BYTES, InputFidelityError
    from remote_input_fidelity import (json_input_parts, ResolvedJSONInputs, decode_json_inputs)

PROTOCOL = "numpy-session-inputs-v1"
CHUNK_BYTES = 512 * 1024
CACHE_THRESHOLD = 1024 * 1024
MAX_REFS = 4096


class SessionInputError(ValueError):
    """Input identity unknown, never an attributable candidate failure."""


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _sha(value):
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise SessionInputError("invalid storage/manifest digest")
    return value


def manifest_body(manifest):
    if not isinstance(manifest, dict) or set(manifest) != {"protocol", "storages", "tensors", "scalars", "hash"}:
        raise SessionInputError("invalid session input manifest fields")
    body = {k: v for k, v in manifest.items() if k != "hash"}
    if body["protocol"] != PROTOCOL or _sha(manifest["hash"]) != digest(body):
        raise SessionInputError("session input protocol/hash mismatch")
    refs = body["storages"]
    if not isinstance(refs, list) or len(refs) > MAX_REFS:
        raise SessionInputError("invalid storage refs")
    total = 0
    sizes = {}
    for ref in refs:
        if not isinstance(ref, dict) or set(ref) != {"sha256", "bytes"}:
            raise SessionInputError("invalid storage ref fields")
        sha, size = _sha(ref["sha256"]), ref["bytes"]
        if type(size) is not int or not 0 <= size <= MAX_STORAGE_BYTES:
            raise SessionInputError("invalid storage size")
        if sha in sizes and sizes[sha] != size:
            raise SessionInputError("same digest with different lengths")
        sizes[sha] = size
        total += size
    if not isinstance(body["tensors"], dict) or not isinstance(body["scalars"], dict):
        raise SessionInputError("invalid tensor/scalar metadata")
    if total + len(json.dumps(body["scalars"], ensure_ascii=True, allow_nan=False).encode()) > MAX_STORAGE_BYTES:
        raise SessionInputError("input storage/scalars exceed byte limit")
    return body


class PreparedInputs(dict):
    """JSON-safe descriptor with immutable raw storage kept off the wire."""
    def __init__(self, metadata, storages):
        body = dict(metadata, protocol=PROTOCOL,
                    storages=[{"sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b)} for b in storages])
        super().__init__(body, hash=digest(body))
        self.raw_storages = tuple(storages)
        manifest_body(self)


def prepare_inputs(inputs):
    metadata, storages = json_input_parts(inputs)
    if sum(map(len, storages)) < CACHE_THRESHOLD:
        return dict(metadata, storages=[base64.b64encode(b).decode("ascii") for b in storages])
    return PreparedInputs(metadata, storages)


class SessionInputs:
    """One upload transaction; complete blobs only become visible on commit.

    LRU eviction excludes the current batch. No filenames, shared state, retry
    of uncertain chunks, or cross-session reuse. Capacity includes reserved
    partial bytes; promotion may temporarily copy bytes once to make immutable.
    """
    def __init__(self, capacity=MAX_STORAGE_BYTES):
        if type(capacity) is not int or not 0 < capacity <= MAX_STORAGE_BYTES:
            raise SessionInputError("invalid cache capacity")
        self.session_id = secrets.token_hex(16)
        self.capacity = capacity
        self.blobs = OrderedDict()
        self.pending = None
        self.ready = None
        self.active = True

    def advertisement(self):
        return {"protocol": PROTOCOL, "session_id": self.session_id,
                "chunk_bytes": CHUNK_BYTES, "capacity": self.capacity,
                "chunk_encodings": ["raw", "zlib"]}

    def _session(self, session_id):
        if not self.active or session_id != self.session_id:
            raise SessionInputError("expired/different input session")

    def _receipt(self, manifest_hash):
        return {"success": True, "session_id": self.session_id, "manifest_hash": manifest_hash}

    def abort(self):
        self.pending = self.ready = None

    def begin(self, session_id, manifest):
        self._session(session_id)
        body = manifest_body(manifest)
        # Deep copy metadata; even an in-process caller cannot rewrite a batch.
        stored = json.loads(json.dumps(manifest, allow_nan=False))
        needed = {r["sha256"]: r["bytes"] for r in body["storages"]}
        if sum(needed.values()) > self.capacity:
            raise SessionInputError("batch exceeds session cache capacity")
        self.abort()
        for sha in needed:
            if sha in self.blobs:
                self.blobs.move_to_end(sha)
        reserved = sum(size for sha, size in needed.items() if sha not in self.blobs)
        while sum(map(len, self.blobs.values())) + reserved > self.capacity:
            victim = next((sha for sha in self.blobs if sha not in needed), None)
            if victim is None:
                raise SessionInputError("cannot reserve bounded input batch")
            del self.blobs[victim]
        missing = [{"sha256": sha, "bytes": size} for sha, size in needed.items() if sha not in self.blobs]
        self.pending = {"manifest": stored, "buffers": {r["sha256"]: bytearray() for r in missing}}
        return dict(self._receipt(stored["hash"]), missing=missing)

    def _batch(self, session_id, manifest_hash):
        self._session(session_id)
        if self.pending is None or manifest_hash != self.pending["manifest"]["hash"]:
            raise SessionInputError("missing/different upload transaction")
        return self.pending

    def chunk(self, session_id, manifest_hash, storage_hash, offset, data, encoding="raw"):
        batch = self._batch(session_id, manifest_hash)
        buf = batch["buffers"].get(storage_hash)
        if buf is None or type(offset) is not int or offset != len(buf):
            raise SessionInputError("unknown blob or noncontiguous/replayed chunk")
        if type(data) is not str or not 0 < len(data) <= 4 * ((CHUNK_BYTES + 2) // 3):
            raise SessionInputError("invalid chunk encoding/size")
        try:
            raw = base64.b64decode(data, validate=True)
        except (ValueError, UnicodeError) as exc:
            raise SessionInputError("invalid chunk base64") from exc
        if encoding == "zlib":
            try:
                decoder = zlib.decompressobj()
                # Never unbounded decompress()/flush(): even hostile compressed
                # input can allocate at most one chunk (+ overflow sentinel).
                raw = decoder.decompress(raw, CHUNK_BYTES + 1)
                if (len(raw) > CHUNK_BYTES or not decoder.eof
                        or decoder.unconsumed_tail or decoder.unused_data):
                    raise SessionInputError("compressed chunk exceeds bounds/incomplete/trailing data")
            except zlib.error as exc:
                raise SessionInputError("invalid compressed chunk") from exc
        elif encoding != "raw":
            raise SessionInputError("unknown chunk encoding")
        size = next(r["bytes"] for r in batch["manifest"]["storages"] if r["sha256"] == storage_hash)
        if not raw or len(raw) > CHUNK_BYTES or offset + len(raw) > size:
            raise SessionInputError("chunk exceeds reserved bytes")
        buf.extend(raw)
        return dict(self._receipt(manifest_hash), storage_hash=storage_hash, next_offset=len(buf))

    def _values(self, manifest, additions=None):
        values = []
        additions = additions or {}
        for ref in manifest["storages"]:
            raw = additions.get(ref["sha256"], self.blobs.get(ref["sha256"]))
            if (type(raw) is not bytes or len(raw) != ref["bytes"]
                    or hashlib.sha256(raw).hexdigest() != ref["sha256"]):
                raise SessionInputError("storage absent/incomplete/digest mismatch")
            values.append(raw)
        return values

    def commit(self, session_id, manifest_hash):
        batch = self._batch(session_id, manifest_hash)
        manifest = batch["manifest"]
        additions = {sha: bytes(buf) for sha, buf in batch["buffers"].items()}
        values = self._values(manifest, additions)
        # Same scalar/view validation as inline JSON, before any candidate runs.
        decode_json_inputs(ResolvedJSONInputs(manifest, values))
        self.blobs.update(additions)
        self.ready = manifest
        self.pending = None
        return self._receipt(manifest_hash)

    def resolve(self, wire):
        if not isinstance(wire, dict):
            raise SessionInputError("invalid input reference")
        self._session(wire.get("session_id"))
        manifest = {k: v for k, v in wire.items() if k != "session_id"}
        manifest_body(manifest)
        if self.ready is None or digest(self.ready) != digest(manifest):
            raise SessionInputError("input batch not committed or manifest drifted")
        values = self._values(manifest)
        return ResolvedJSONInputs(manifest, values), {
            "protocol": PROTOCOL, "session_id": self.session_id,
            "manifest_hash": manifest["hash"], "storage_bytes": sum(map(len, values)),
            "storage_count": len(values), "content_verified": True}

    def close(self):
        self.abort()
        self.blobs.clear()
        self.active = False
