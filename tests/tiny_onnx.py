"""Write a minimal valid ONNX model by hand (the `onnx` package is not needed).

Model: input (batch, 3, H, W) -> mean over H and W -> (batch, 3) -> matmul with a 3 x K weight
matrix -> logits (batch, K). Good enough to test the real ONNX Runtime loading path.
"""

import struct


def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        byte = n & 0x7F
        n >>= 7
        out.append(byte | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _tag(field: int, wire: int) -> bytes:
    return _varint((field << 3) | wire)


def _int(field: int, value: int) -> bytes:
    return _tag(field, 0) + _varint(value)


def _bytes(field: int, payload: bytes) -> bytes:
    return _tag(field, 2) + _varint(len(payload)) + payload


def _str(field: int, text: str) -> bytes:
    return _bytes(field, text.encode())


def _value_info(name: str, dims: list) -> bytes:
    shape = b""
    for d in dims:
        shape += _bytes(1, _int(1, d) if isinstance(d, int) else _str(2, d))
    tensor_type = _int(1, 1) + _bytes(2, shape)  # elem_type FLOAT, shape
    return _str(1, name) + _bytes(2, _bytes(1, tensor_type))


def _node(op: str, inputs: list[str], output: str, attributes: bytes = b"") -> bytes:
    body = b"".join(_str(1, i) for i in inputs) + _str(2, output) + _str(4, op) + attributes
    return _bytes(1, body)


def tiny_model(weights: list[list[float]], input_dims=("batch", 3, "h", "w")) -> bytes:
    """`weights` is a 3 x K matrix (list of 3 rows)."""
    k = len(weights[0])
    axes = _bytes(5, _str(1, "axes") + _int(20, 7) + _int(8, 2) + _int(8, 3))
    keepdims = _bytes(5, _str(1, "keepdims") + _int(20, 2) + _int(3, 0))
    raw = b"".join(struct.pack("<f", v) for row in weights for v in row)
    weight = _bytes(5, _int(1, 3) + _int(1, k) + _int(2, 1) + _str(8, "w") + _bytes(9, raw))
    graph = (
        _node("ReduceMean", ["input"], "pooled", axes + keepdims)
        + _node("MatMul", ["pooled", "w"], "logits")
        + _str(2, "tiny")
        + weight
        + _bytes(11, _value_info("input", list(input_dims)))
        + _bytes(12, _value_info("logits", ["batch", k]))
    )
    opset = _bytes(8, _str(1, "") + _int(2, 13))
    return _int(1, 8) + _str(2, "agriscan-tests") + _bytes(7, graph) + opset
