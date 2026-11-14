"""Host slot packing primitives; the Atlas launcher does not yet invoke these.

Buffers contain already encoded BF16 or E4M3FN elements in row-major order.
A bias slot [1, N] from a rank-one host vector repeats its row for Pass 7's
broadcast erasure. Output unpacking also widens BF16 device elements to f32
when requested by the manifest. No device transfer occurs here.
"""
import re

# What the host writes into the padded lanes of an input tile, by the pad tag's neutral and element size (little
# endian): BF16 0x0000, 0xFF80 (-inf), 0x3F80 (1.0), 0x7F80 (+inf); E4M3FN 0x00 and 0x38 (1.0, which has no infinity).
_FILLS = {
    None: {1: b"\x00", 2: b"\x00\x00"},
    "zero": {1: b"\x00", 2: b"\x00\x00"},
    "neg_inf": {2: b"\x80\xff"},
    "one": {1: b"\x38", 2: b"\x80\x3f"},
    "pos_inf": {2: b"\x80\x7f"},
}


def _view(slot, host_shape):
    dtype = slot["host_dtype"]
    if dtype not in ("bf16", "f8E4M3FN"):
        raise ValueError("packing requires already encoded BF16 or E4M3FN")
    size = 2 if dtype == "bf16" else 1
    rows, cols = slot["logical_shape"]
    if not (1 <= rows <= 32 and 1 <= cols <= 32):
        raise ValueError("logical shape must fit one tile")
    match = re.fullmatch(r"affine_map<\(\)\[s0, s1, s2, s3, s4, s5\] -> \((\d+), (\d+)\)>", slot["tile_origin"])
    if not match:
        raise ValueError("packing requires constant nonnegative origins")
    row, col = map(int, match.groups())
    bias = len(host_shape) == 1
    if bias:
        if slot["role"] != "bias_bf16" or dtype != "bf16" or rows != 1 or row != 0:
            raise ValueError("rank-one buffers require a BF16 bias row")
        height, width = 1, host_shape[0]
    elif len(host_shape) == 2:
        height, width = host_shape
    else:
        raise ValueError("host shape must have rank one or two")
    if min(height, width) <= 0 or row + rows > height or col + cols > width:
        raise ValueError("logical tile exceeds host buffer shape")
    return size, rows, cols, row, col, height, width, bias


def pack_input_tile(host, host_shape, slot):
    """Return 1024 encoded elements, filling invalid lanes with Pass 4's neutral."""
    size, rows, cols, row, col, height, width, bias = _view(slot, host_shape)
    if len(host) != height * width * size:
        raise ValueError("host buffer length disagrees with shape and dtype")
    neutral = slot.get("pad_neutral")
    if neutral not in (None, *_FILLS):
        raise ValueError("unsupported padding neutral")
    if (rows < 32 or cols < 32) and not bias and neutral is None:
        raise ValueError("partial input requires Pass 4 padding neutral")
    if neutral in ("neg_inf", "pos_inf") and size != 2:
        raise ValueError("E4M3FN has no infinity")
    fill = _FILLS[neutral][size]
    result = bytearray(fill * 1024)
    for r in range(32 if bias else rows):
        start = ((row + (0 if bias else r)) * width + col) * size
        result[r * 32 * size:(r * 32 + cols) * size] = host[start:start + cols * size]
    return bytes(result)


def unpack_output_tile(tile, host, host_shape, slot):
    """Crop a BF16 device tile, optionally widening its little-endian bits to f32."""
    if slot["role"] != "output_bf16" or slot["host_dtype"] not in ("bf16", "f32"):
        raise ValueError("outputs require BF16 device slots with BF16 or f32 host dtype")
    _, rows, cols, row, col, height, width, _ = _view({**slot, "host_dtype": "bf16"}, host_shape)
    size = 4 if slot["host_dtype"] == "f32" else 2
    if len(tile) != 2048 or len(host) != height * width * size:
        raise ValueError("output buffer length disagrees with shape and dtype")
    for r in range(rows):
        start = ((row + r) * width + col) * size
        source = tile[r * 64:r * 64 + cols * 2]
        if size == 4:
            source = b"".join(b"\x00\x00" + source[c:c + 2] for c in range(0, len(source), 2))
        host[start:start + cols * size] = source
