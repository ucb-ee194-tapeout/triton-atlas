"""Host packer: moves tiles between host tensors and the device's DRAM slots, from the kernel metadata's slot list.

A slot is a 32x32 tile (a vector slot is one lane per row or column, repeated across the tile). Per slot:
memory inside `valid` is gathered from the host tensor with both strides, `other` (0 when there is none) fills the
rest of `block`, and everything outside `block` is 0, all in the slot's tile dtype. A constant slot is its raw
`pattern`. A transposed slot is written transposed. `block` and `valid` are [lo_row, lo_col, hi_row, hi_col] in
tile-local coordinates; a vector slot's lanes run from lo_row to hi_row of them (lane i at offset + i * row_stride).

Bytes: FP8 tiles are row-major (1024 bytes); BF16 tiles use atlas-mlir's split-half order (2048 bytes): element
(row, col) at byte (col // 16) * 1024 + (row * 16 + col % 16) * 2. Slot i sits at base + i * SLOT_BYTES.

Host buffers hold encoded elements, little endian: bf16 and f8E4M3FN raw bits, or f32. Only the casts the compiler
folds into a slot exist: the same dtype on input (f32 inputs are rejected by the compiler), BF16 to f32 on output.
"""
SCHEMA_VERSION = 2
SLOT_BYTES = 2048
_SIZE = {"bf16": 2, "f8E4M3FN": 1, "f32": 4}


def _check_version(manifest):
    if manifest.get("version") != SCHEMA_VERSION:
        raise ValueError(f"packer needs a version-{SCHEMA_VERSION} kernel manifest, got {manifest.get('version')!r}")


def _host_size(dtype):
    if dtype not in _SIZE:
        raise ValueError(f"unsupported host dtype {dtype!r}")
    return _SIZE[dtype]


def _to_tile_bits(host, index, host_dtype, tile_dtype):
    """The raw element at `index` of a host buffer, as raw bits of the tile dtype."""
    size = _host_size(host_dtype)
    raw = int.from_bytes(host[index * size:(index + 1) * size], "little")
    if host_dtype == tile_dtype:
        return raw
    raise ValueError(f"unsupported cast from host {host_dtype} to tile {tile_dtype}")


def _fill(tile, lo_row, lo_col, hi_row, hi_col, value):
    for row in range(lo_row, hi_row):
        tile[row * 32 + lo_col:row * 32 + hi_col] = [value] * (hi_col - lo_col)


def build_tile(slot, host):
    """The slot's logical 32x32 tile as 1024 raw values (row-major), before byte ordering."""
    if slot["kind"] == "constant":
        return list(slot["pattern"])
    size = _host_size(slot["host_dtype"])
    if len(host) % size:
        raise ValueError("host buffer length disagrees with its dtype")
    count = len(host) // size
    tile = [0] * 1024
    block, valid = slot["block"], slot["valid"]
    _fill(tile, *block, slot["other"] or 0)
    offset, row_stride, col_stride = slot["offset"], slot["row_stride"], slot["col_stride"]

    def read(index):
        if not 0 <= index < count:
            raise ValueError("slot reads outside its host buffer")
        return _to_tile_bits(host, index, slot["host_dtype"], slot["tile_dtype"])

    layout = slot["vector_layout"]
    if layout == "none":
        for row in range(valid[0], valid[2]):
            for col in range(valid[1], valid[3]):
                tile[row * 32 + col] = read(offset + row * row_stride + col * col_stride)
    else:
        lanes = [slot["other"] or 0 if block[0] <= i < block[2] else 0 for i in range(32)]
        for i in range(valid[0], valid[2]):
            lanes[i] = read(offset + i * row_stride)
        for row in range(32):
            for col in range(32):
                tile[row * 32 + col] = lanes[row if layout == "rows" else col]
    if slot["transposed"]:
        tile = [tile[col * 32 + row] for row in range(32) for col in range(32)]
    return tile


def _bf16_position(row, col):
    return (col // 16) * 1024 + (row * 16 + col % 16) * 2


def pack_input_tile(slot, host):
    """The bytes of one input slot: 1024 for an FP8 tile, 2048 for a BF16 tile."""
    tile = build_tile(slot, host)
    if slot["tile_dtype"] == "f8E4M3FN":
        return bytes(tile)
    out = bytearray(2048)
    for row in range(32):
        for col in range(32):
            position = _bf16_position(row, col)
            out[position:position + 2] = tile[row * 32 + col].to_bytes(2, "little")
    return bytes(out)


def unpack_output_tile(slot, data, host):
    """Write the valid rectangle of one BF16 output slot's bytes into a mutable host buffer, widened to its dtype."""
    if slot["tile_dtype"] != "bf16" or slot["host_dtype"] not in ("bf16", "f32"):
        raise ValueError("outputs are BF16 tiles written to a bf16 or f32 host tensor")
    if len(data) < 2048:
        raise ValueError("output slot data must be 2048 bytes")
    size = _host_size(slot["host_dtype"])
    count = len(host) // size
    valid, layout = slot["valid"], slot["vector_layout"]

    def element(row, col):
        position = _bf16_position(row, col)
        bits = int.from_bytes(data[position:position + 2], "little")
        return bits if size == 2 else bits << 16

    def write(index, value):
        if not 0 <= index < count:
            raise ValueError("slot writes outside its host buffer")
        host[index * size:(index + 1) * size] = value.to_bytes(size, "little")

    offset, row_stride, col_stride = slot["offset"], slot["row_stride"], slot["col_stride"]
    if layout == "none":
        for row in range(valid[0], valid[2]):
            for col in range(valid[1], valid[3]):
                write(offset + row * row_stride + col * col_stride, element(row, col))
    else:
        for i in range(valid[0], valid[2]):
            write(offset + i * row_stride, element(i, 0) if layout == "rows" else element(0, i))


def pack_inputs(manifest, host):
    """The DRAM window of all input slots; `host` maps a kernel argument index to its buffer."""
    _check_version(manifest)
    window = bytearray(len(manifest["inputs"]) * SLOT_BYTES)
    for slot in manifest["inputs"]:
        if slot["kind"] == "memory" and slot["arg"] not in host:
            raise ValueError(f"no host buffer for kernel argument {slot['arg']}")
        data = pack_input_tile(slot, host.get(slot.get("arg")))
        window[slot["index"] * SLOT_BYTES:slot["index"] * SLOT_BYTES + len(data)] = data
    return bytes(window)


def unpack_outputs(manifest, window, host):
    """Write every output slot's valid rectangle from the output window into the host buffers (bytearrays)."""
    _check_version(manifest)
    for slot in manifest["outputs"]:
        if slot["arg"] not in host:
            raise ValueError(f"no host buffer for kernel argument {slot['arg']}")
        unpack_output_tile(slot, window[slot["index"] * SLOT_BYTES:(slot["index"] + 1) * SLOT_BYTES], host[slot["arg"]])
