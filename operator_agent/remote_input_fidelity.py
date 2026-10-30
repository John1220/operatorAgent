"""Exact JSON transport and Torch storage/view readback for remote L2.

Deployed as a standalone sibling of run_kernel.py, not a copied implementation.
No unpickling on the wire. This is fidelity evidence, not hostile-code isolation.
"""
from __future__ import annotations

import base64
import hashlib
import struct

import numpy as np

try:
    from . import input_fidelity as fidelity
except ImportError:  # deployed runner imports sibling modules
    import input_fidelity as fidelity

InputFidelityError = fidelity.InputFidelityError
JSON_PROTOCOL = "numpy-storage-json-v1"


def _unbase64(data, remaining):
    if type(data) is not str or len(data) > 4 * ((remaining + 2) // 3):
        raise InputFidelityError("base64 类型或字节上限无效")
    try:
        raw = base64.b64decode(data, validate=True)
    except (ValueError, UnicodeError) as exc:
        raise InputFidelityError("base64 无效") from exc
    if len(raw) > remaining:
        raise InputFidelityError("JSON 输入超过字节上限")
    return raw


def _scalar_wire(value):
    kind = type(value).__name__
    if isinstance(value, np.generic):
        return {"kind": "numpy", "dtype": value.dtype.str,
                "data": base64.b64encode(value.tobytes()).decode("ascii")}
    if type(value) in (type(None), bool, int, str):
        return {"kind": kind, "value": value}
    raw = (struct.pack("!d", value) if type(value) is float else
           struct.pack("!dd", value.real, value.imag) if type(value) is complex else value)
    return {"kind": kind, "data": base64.b64encode(raw).decode("ascii")}


def _scalar_from_wire(spec):
    if not isinstance(spec, dict):
        raise InputFidelityError("标量描述无效")
    kind = spec.get("kind")
    if type(kind) is not str:
        raise InputFidelityError("标量 kind 必须是字符串")
    native = {"NoneType": type(None), "bool": bool, "int": int, "str": str}
    if kind in native:
        value = spec.get("value")
        if type(value) is not native[kind]:
            raise InputFidelityError("标量类型与值不一致")
        return value
    raw = _unbase64(spec.get("data"), fidelity.MAX_STORAGE_BYTES)
    if kind == "bytes":
        return raw
    if kind == "float" and len(raw) == 8:
        return struct.unpack("!d", raw)[0]
    if kind == "complex" and len(raw) == 16:
        return complex(*struct.unpack("!dd", raw))
    if kind == "numpy":
        try:
            dtype = np.dtype(spec["dtype"])
            if dtype.kind in "biufc" and len(raw) == dtype.itemsize:
                return np.frombuffer(raw, dtype=dtype)[0]
        except (KeyError, TypeError, ValueError) as exc:
            raise InputFidelityError("NumPy 标量 dtype 无效") from exc
    raise InputFidelityError("未知标量类型或字节长度无效")


def json_input_parts(inputs):
    """One scalar/size authority for inline JSON and session-owned storage."""
    packet = fidelity.encode_inputs(inputs)
    # Scalars are bounded too; floats/nonfinite values use bytes, not lossy JSON.
    import json
    scalars = {k: _scalar_wire(v) for k, v in packet["scalars"].items()}
    scalar_bytes = len(json.dumps(scalars, ensure_ascii=True, allow_nan=False).encode())
    if sum(map(len, packet["storages"])) + scalar_bytes > fidelity.MAX_STORAGE_BYTES:
        raise InputFidelityError("JSON 输入超过字节上限")
    return {"protocol": JSON_PROTOCOL, "tensors": packet["tensors"], "scalars": scalars}, packet["storages"]


def encode_json_inputs(inputs):
    metadata, storages = json_input_parts(inputs)
    return dict(metadata, storages=[base64.b64encode(b).decode("ascii") for b in storages])


class ResolvedJSONInputs(dict):
    """In-process server carrier, impossible to construct via JSON fields alone.

    Cached immutable bytes avoid remote re-base64/decoding. The same size,
    scalar and view decoder still runs; each invocation gets fresh mutable roots.
    """
    def __init__(self, metadata, storages):
        super().__init__(metadata, protocol=JSON_PROTOCOL)
        self.raw_storages = tuple(storages)


def decode_json_inputs(payload):
    import json
    if not isinstance(payload, dict) or payload.get("protocol") != JSON_PROTOCOL:
        raise InputFidelityError("未知 JSON 输入保真协议")
    scalars = payload.get("scalars")
    resolved = isinstance(payload, ResolvedJSONInputs)
    storages = list(payload.raw_storages) if resolved else payload.get("storages")
    if not isinstance(scalars, dict) or not isinstance(storages, list):
        raise InputFidelityError("JSON 输入载荷无效")
    remaining = fidelity.MAX_STORAGE_BYTES - len(json.dumps(
        scalars, ensure_ascii=True, allow_nan=False).encode())
    if remaining < 0:
        raise InputFidelityError("JSON 标量超过字节上限")
    decoded = []
    for data in storages:
        if resolved:
            if type(data) is not bytes or len(data) > remaining:
                raise InputFidelityError("cached storage 类型或字节上限无效")
            raw = data
        else:
            raw = _unbase64(data, remaining)
        decoded.append(raw)
        remaining -= len(raw)
    return fidelity.decode_inputs({"protocol": fidelity.PROTOCOL, "storages": decoded,
        "tensors": payload.get("tensors"),
        "scalars": {k: _scalar_from_wire(v) for k, v in scalars.items()}})


def to_torch_inputs(inputs, device):
    """Copy whole storage once per group, then set exact typed device views.

    Never .contiguous() or precision-cast an individual input. Torch cannot
    express negative strides or NumPy readonly semantics: refuse, don't pretend.
    """
    import torch
    packet = fidelity.encode_inputs(inputs, limit=fidelity.MAX_SEED_STORAGE_BYTES)
    specs = []
    for name, spec in packet["tensors"].items():
        dtype = np.dtype(spec["dtype"])
        if not spec["writeable"]:
            raise InputFidelityError("Torch 无只读属性，不能保真 readonly 输入")
        if any(s < 0 for s in spec["strides"]):
            raise InputFidelityError("Torch 不支持负 stride，不转连续副本")
        if spec["offset"] % dtype.itemsize or any(s % dtype.itemsize for s in spec["strides"]):
            raise InputFidelityError("Torch view 需要按 dtype 对齐的 stride/offset")
        try:
            torch_dtype = torch.from_numpy(np.empty(0, dtype=dtype)).dtype
        except (TypeError, ValueError) as exc:
            raise InputFidelityError(f"Torch 不支持 dtype {dtype}") from exc
        specs.append((name, spec, dtype.itemsize, torch_dtype))
    buffers = [torch.from_numpy(np.frombuffer(bytearray(b), dtype=np.uint8)).to(device)
               for b in packet["storages"]]
    out = dict(packet["scalars"])
    for name, spec, itemsize, dtype in specs:
        out[name] = torch.empty(0, dtype=dtype, device=device).set_(
            buffers[spec["storage"]].untyped_storage(), spec["offset"] // itemsize,
            tuple(spec["shape"]), tuple(s // itemsize for s in spec["strides"]))
    return out


def torch_input_readback(inputs):
    """Read actual device tensors and storage, NOT transport declarations."""
    import torch
    observed, groups, devices = {}, {}, {}
    total_bytes = 0
    for name in sorted(inputs):
        value = inputs[name]
        if not isinstance(value, torch.Tensor):
            if isinstance(value, np.ndarray):
                raise InputFidelityError("Torch 候选实际输入仍是 NumPy，不能当成设备 tensor 回读")
            observed[name] = fidelity.describe_inputs({name: value})[name]
            continue
        if value.layout != torch.strided:
            raise InputFidelityError("只支持 strided Torch tensor")
        storage = value.untyped_storage()
        identity = (str(value.device), storage.data_ptr(), storage.nbytes())
        if identity not in groups:
            total_bytes += storage.nbytes()
            if total_bytes > fidelity.MAX_SEED_STORAGE_BYTES:
                raise InputFidelityError("Torch storage 超过字节上限")
            raw_view = torch.empty(0, dtype=torch.uint8, device=value.device).set_(
                storage, 0, (storage.nbytes(),), (1,))
            raw = raw_view.cpu().numpy().tobytes()
            groups[identity] = (len(groups), hashlib.sha256(raw).hexdigest())
        group, digest = groups[identity]
        if value.numel() * value.element_size() > fidelity.MAX_SEED_STORAGE_BYTES:
            raise InputFidelityError("Torch logical tensor 超过字节上限")
        logical = value.detach().cpu().numpy()
        observed[name] = {"shape": list(value.shape), "dtype": logical.dtype.str,
            "strides": [s * value.element_size() for s in value.stride()],
            "offset": value.storage_offset() * value.element_size(), "storage": group,
            # Torch tensors are writable; never echo NumPy's readonly declaration.
            "writeable": True, "storage_bytes": storage.nbytes(),
            "storage_digest": digest,
            "value_digest": hashlib.sha256(logical.tobytes(order="C")).hexdigest()}
        devices[name] = str(value.device)
    return {"protocol": fidelity.PROTOCOL, "representation": "torch-storage-v1",
            "observed": observed, "tensor_devices": devices}


def torch_output_digest(output):
    """Full actual dtype/shape/bytes fingerprint before fp32 comparison casting."""
    import torch
    value = output.detach().contiguous()
    digest = hashlib.sha256(str((str(value.dtype), tuple(value.shape))).encode())
    digest.update(value.reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


# ── 按 seed 物化输入（2026-10-20）───────────────────────────────────────────
#
# 为什么要有这条协议: L2 的输入原先整份 base64 传过去，fp16 4096³ 两个张量 ≈ 90 MB 文本；
# 实测链路 ~0.28 MB/s ⇒ 一次 L2 光传输就 570 s（见评估文档 §9.8.1）。契约本来就拥有
# `seed`/`shape`/`dtype`，物化又是确定性的 —— 于是可以只传描述 + digest。
#
# **判据不许因此变松**：控制端先在自己的机器上物化、比对 digest；远端再物化一次、
# 再比一次。任意一侧不一致就退回「传字节」那条老路（`input_seed_mismatch`）。
SEED_PROTOCOL = "numpy-seed-v1"

_NP_DTYPES = {"fp32": "float32", "fp16": "float16", "fp64": "float64",
              "int32": "int32", "int64": "int64"}


def seed_dtype(name):
    """case 里写的 dtype（`fp16`）→ numpy dtype；表外一律拒绝（不降级）。"""
    key = str(name)
    if key not in _NP_DTYPES:
        raise InputFidelityError(f"seed 物化不支持 dtype={key!r}")
    return np.dtype(_NP_DTYPES[key])


def materialize_seed_inputs(spec):
    """按 `{"seed": int, "tensors": [{"name","shape","dtype"}]}` 造输入。

    抽取顺序**必须**与 `bench_cases.BenchmarkCase.materialize` 一致：一个 rng、
    按声明顺序、浮点 `standard_normal`、整数 `integers(1,8)`。两个实现分家之日就是
    「控制端以为在验 A、远端实际在验 B」之时，所以那边现在直接调这里。
    """
    if not isinstance(spec, dict) or spec.get("protocol") != SEED_PROTOCOL:
        raise InputFidelityError("未知 seed 物化协议")
    seed = spec.get("seed")
    if type(seed) is not int or seed < 0:
        raise InputFidelityError("seed 必须是非负 int")
    tensors = spec.get("tensors")
    if not isinstance(tensors, list) or not tensors:
        raise InputFidelityError("tensors 不能为空")
    rng = np.random.default_rng(seed)
    out = {}
    for t in tensors:
        if not isinstance(t, dict):
            raise InputFidelityError("tensor 声明必须是对象")
        name, shape = t.get("name"), t.get("shape")
        if not isinstance(name, str) or not name:
            raise InputFidelityError("tensor 缺 name")
        if not isinstance(shape, list) or any(type(d) is not int or d < 0 for d in shape):
            raise InputFidelityError(f"tensor {name!r} 的 shape 无效")
        dt = seed_dtype(t.get("dtype"))
        if dt.kind == "f":
            out[name] = rng.standard_normal(tuple(shape)).astype(dt)
        else:
            out[name] = rng.integers(1, 8, size=tuple(shape)).astype(dt)
    return out


def seed_input_digest(values):
    """输入内容的规范摘要（名字 + dtype + shape + 连续字节），跨机器可复算。"""
    h = hashlib.sha256()
    for name in sorted(values):
        arr = np.ascontiguousarray(values[name])
        h.update(name.encode("utf-8"))
        h.update(b"\0")
        h.update(arr.dtype.str.encode("ascii"))
        h.update(b"\0")
        h.update(repr(tuple(arr.shape)).encode("ascii"))
        h.update(b"\0")
        h.update(arr.tobytes())
        h.update(b"\0")
    return h.hexdigest()
