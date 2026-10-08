"""Bounded NumPy storage/view transport and pre-candidate readback evidence.

Pickling ndarray alone copies logical values, erasing strides/offsets. This
protocol sends each backing storage once and reconstructs views over it. It is
not a hostile-code sandbox or a proof of aliasing/in-place operator semantics.
"""
from __future__ import annotations

import hashlib
import math
import pickle
from typing import Mapping

import numpy as np

PROTOCOL = "numpy-storage-v1"
MAX_STORAGE_BYTES = 256 * 1024 * 1024


class InputFidelityError(ValueError):
    """Cannot preserve these inputs; do not silently make contiguous copies."""


def _storage(array):
    if type(array) is not np.ndarray or array.dtype.kind not in "biufc":
        raise InputFidelityError("只支持无对象的数值 ndarray（不降级 subclass/object dtype）")
    root = array
    while isinstance(root.base, np.ndarray):
        root = root.base
    if (type(root) is not np.ndarray or root.dtype.hasobject
            or (root.base is not None and not isinstance(root.base, (bytes, bytearray, memoryview)))):
        raise InputFidelityError("backing storage 来源不可确立（不读取 stride_tricks/未知对象指针）")
    if not (root.flags.c_contiguous or root.flags.f_contiguous):
        raise InputFidelityError("无法定位 contiguous backing storage，不猜 offset")
    start = int(root.__array_interface__["data"][0])
    offset = int(array.__array_interface__["data"][0]) - start
    _validate_view(array.shape, array.strides, offset, array.dtype, root.nbytes)
    return root, offset, (start, int(root.nbytes))


def _validate_view(shape, strides, offset, dtype, storage_bytes):
    if len(shape) != len(strides) or type(offset) is not int:
        raise InputFidelityError("shape/strides/offset 声明无效")
    if any(type(d) is not int or d < 0 for d in shape):
        raise InputFidelityError("shape 必须是非负整数")
    if any(type(s) is not int for s in strides):
        raise InputFidelityError("strides 必须是字节整数")
    if dtype.kind not in "biufc" or dtype.hasobject:
        raise InputFidelityError("不支持对象/结构化 dtype")
    if math.prod(shape) * dtype.itemsize > MAX_STORAGE_BYTES:
        raise InputFidelityError("logical tensor 超过输入字节上限")
    if any(d == 0 for d in shape):
        low = high = offset
    else:
        low = offset + sum(min(0, (d - 1) * s) for d, s in zip(shape, strides))
        high = offset + sum(max(0, (d - 1) * s) for d, s in zip(shape, strides)) + dtype.itemsize
    if low < 0 or high > storage_bytes:
        raise InputFidelityError("view 越出 backing storage（拒绝越界重建）")


def _scalar(value):
    if type(value) in (type(None), bool, int, float, complex, str, bytes):
        return True
    return isinstance(value, np.generic) and value.dtype.kind in "biufc"


def _describe(inputs, include_storage):
    if not isinstance(inputs, Mapping) or any(type(k) is not str for k in inputs):
        raise InputFidelityError("输入必须是字符串键映射")
    storages, tensors, scalars, observed = [], {}, {}, {}
    groups = {}
    size = 0
    for name in sorted(inputs):
        value = inputs[name]
        if isinstance(value, np.ndarray):
            root, offset, identity = _storage(value)
            if identity not in groups:
                size += int(root.nbytes)
                if size > MAX_STORAGE_BYTES:
                    raise InputFidelityError("backing storage 总量超过输入字节上限")
                data = root.tobytes(order="A")
                groups[identity] = (len(groups), hashlib.sha256(data).hexdigest())
                if include_storage:
                    storages.append(data)
            group, storage_hash = groups[identity]
            spec = {"shape": list(value.shape), "dtype": value.dtype.str,
                    "strides": list(value.strides), "offset": offset,
                    "storage": group, "writeable": bool(value.flags.writeable)}
            tensors[name] = spec
            observed[name] = {**spec, "storage_bytes": int(root.nbytes),
                              "storage_digest": storage_hash,
                              "value_digest": hashlib.sha256(value.tobytes(order="C")).hexdigest()}
        elif _scalar(value):
            scalars[name] = value
            observed[name] = {"kind": "scalar", "type": type(value).__name__,
                              "value_digest": hashlib.sha256(pickle.dumps(value, protocol=4)).hexdigest()}
        else:
            raise InputFidelityError(f"输入 {name!r} 不是可保真的数值 ndarray/标量")
    return {"protocol": PROTOCOL, "storages": storages, "tensors": tensors,
            "scalars": scalars}, observed


def encode_inputs(inputs):
    return _describe(inputs, True)[0]


def describe_inputs(inputs):
    """Read actual values/layout; never echo a transport declaration as evidence."""
    return _describe(inputs, False)[1]


def decode_inputs(payload):
    if not isinstance(payload, Mapping) or payload.get("protocol") != PROTOCOL:
        raise InputFidelityError("未知输入保真协议")
    storages = payload.get("storages")
    tensors = payload.get("tensors")
    scalars = payload.get("scalars")
    if (not isinstance(storages, list) or not isinstance(tensors, dict)
            or not isinstance(scalars, dict) or set(tensors) & set(scalars)):
        raise InputFidelityError("输入保真载荷结构无效")
    if any(type(b) is not bytes for b in storages) or sum(map(len, storages)) > MAX_STORAGE_BYTES:
        raise InputFidelityError("storage 字节类型或总量无效")
    # Views share the same u8 root; preserve cross-input storage relationships.
    buffers = [np.frombuffer(bytearray(b), dtype=np.uint8) for b in storages]
    out = dict(scalars)
    if any(type(k) is not str for k in out) or any(not _scalar(v) for v in out.values()):
        raise InputFidelityError("标量载荷无效")
    for name, spec in tensors.items():
        try:
            if type(name) is not str or not isinstance(spec, dict):
                raise InputFidelityError("tensor 载荷无效")
            group = spec["storage"]
            if type(group) is not int or not 0 <= group < len(buffers):
                raise InputFidelityError("storage 索引无效")
            shape, strides = spec["shape"], spec["strides"]
            if not isinstance(shape, list) or not isinstance(strides, list):
                raise InputFidelityError("shape/strides 必须是列表")
            dtype, offset = np.dtype(spec["dtype"]), spec["offset"]
            _validate_view(shape, strides, offset, dtype, len(storages[group]))
            if type(spec["writeable"]) is not bool:
                raise InputFidelityError("writeable 必须是 bool")
            view = np.ndarray(tuple(shape), dtype=dtype, buffer=buffers[group],
                              offset=offset, strides=tuple(strides))
            view.setflags(write=spec["writeable"])
            out[name] = view
        except (KeyError, TypeError, ValueError) as exc:
            raise InputFidelityError(f"输入 {name!r} 无法重建: {exc}") from exc
    return out


def check_readback(expected, report):
    """Three-state observed fidelity: absent/invalid=None, mismatch=False, exact=True."""
    metrics = {"input_fidelity_expected": expected, "input_fidelity_verified": None}
    if not isinstance(report, dict) or report.get("protocol") != PROTOCOL:
        metrics["input_fidelity_reason"] = "缺少执行侧输入回读证据或协议不兼容"
        return metrics
    actual = report.get("observed")
    metrics["input_fidelity"] = report
    if not isinstance(actual, dict):
        metrics["input_fidelity_reason"] = "执行侧输入回读载荷无效"
        return metrics
    ok = _typed_equal(actual, expected)
    metrics["input_fidelity_verified"] = ok
    metrics["input_fidelity_reason"] = ("执行侧值/dtype/shape/strides/offset/storage/只读属性全部一致"
                                         if ok else "执行侧实际输入与声明不一致，不能确立正确性")
    return metrics


def _typed_equal(left, right):
    # Python equality conflates 1/True and int/float. Runtime layout evidence
    # must preserve field types as well as values, rather than accepting that.
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_typed_equal(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(_typed_equal(a, b) for a, b in zip(left, right))
    return left == right
