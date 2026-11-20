"""The host packer: slot tiles from host tensors, byte order, and outputs (kernel metadata v2)."""
import struct
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "third_party/atlas-mlir/python"))
# Import the module itself: the package's __init__ needs the Triton plugin, which the packer does not.
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("boundary", ROOT / "python/triton_atlas_backend/boundary.py")
boundary = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(boundary)
from atlas_mlir_native.layout import pack_bf16_pair_halves, unpack_bf16_pair_halves  # noqa: E402

NEG_INF = 0xFF80


def slot(**kw):
    base = dict(index=0, kind="memory", arg=0, role="activation_bf16", offset=0, row_stride=32, col_stride=1,
                block=[0, 0, 32, 32], valid=[0, 0, 32, 32], other=None, vector_layout="none", transposed=False,
                host_dtype="bf16", tile_dtype="bf16")
    base.update(kw)
    return base


def bf16_buffer(values):
    return b"".join(v.to_bytes(2, "little") for v in values)


def logical(data):
    """Row-major 32x32 values from a BF16 slot's bytes, through atlas-mlir's own helper."""
    row_major = unpack_bf16_pair_halves(data)
    return [int.from_bytes(row_major[i:i + 2], "little") for i in range(0, 2048, 2)]


class PackerTest(unittest.TestCase):

    def test_bf16_split_half_order_matches_atlas_mlir(self):
        host = bf16_buffer(range(1024))
        data = boundary.pack_input_tile(slot(), host)
        self.assertEqual(data, pack_bf16_pair_halves(host))
        self.assertEqual(logical(data), list(range(1024)))

    def test_fp8_stays_row_major(self):
        host = bytes(range(256)) * 4
        data = boundary.pack_input_tile(slot(host_dtype="f8E4M3FN", tile_dtype="f8E4M3FN"), host)
        self.assertEqual(data, host)

    def test_valid_other_and_zero_regions(self):
        # 32 rows x 20 cols valid (stride 20 in the host), a block of 24 columns whose last 4 hold `other`.
        host = bf16_buffer([7] * (32 * 20))
        tile = logical(
            boundary.pack_input_tile(slot(row_stride=20, block=[0, 0, 32, 24], valid=[0, 0, 32, 20], other=NEG_INF),
                                     host))
        for row in range(32):
            for col in range(32):
                expected = 7 if col < 20 else NEG_INF if col < 24 else 0
                self.assertEqual(tile[row * 32 + col], expected, (row, col))
        # No `other`: masked lanes are 0.
        tile = logical(boundary.pack_input_tile(slot(row_stride=20, block=[0, 0, 32, 24], valid=[0, 0, 32, 20]), host))
        self.assertEqual(tile[20], 0)

    def test_offset_and_both_strides(self):
        host = bf16_buffer(range(100 * 64))
        tile = logical(boundary.pack_input_tile(slot(offset=5 * 64 + 9, row_stride=64, col_stride=1), host))
        self.assertEqual(tile[0], 5 * 64 + 9)
        self.assertEqual(tile[33], 6 * 64 + 10)
        tile = logical(boundary.pack_input_tile(slot(offset=3, row_stride=1, col_stride=100), host))
        self.assertEqual(tile[1], 3 + 100)
        self.assertEqual(tile[32], 4)

    def test_transposed(self):
        host = bf16_buffer(range(1024))
        tile = logical(boundary.pack_input_tile(slot(transposed=True), host))
        self.assertEqual(tile[1 * 32 + 0], 1)
        self.assertEqual(tile[0 * 32 + 1], 32)

    def test_constant_slot_is_raw_bits(self):
        pattern = [0x7FFF if i % 2 else 0xFFFF for i in range(1024)]
        tile = logical(boundary.pack_input_tile(dict(slot(kind="constant", pattern=pattern), arg=None), None))
        self.assertEqual(tile, pattern)

    def test_vector_layouts(self):
        host = bf16_buffer(range(100, 120))
        rows = logical(
            boundary.pack_input_tile(slot(row_stride=1, vector_layout="rows", block=[0, 0, 20, 1], valid=[0, 0, 20, 1]),
                                     host))
        self.assertEqual([rows[r * 32 + c] for r in (0, 19) for c in (0, 31)], [100, 100, 119, 119])
        self.assertEqual(rows[20 * 32], 0)  # lane 20 is outside the block
        cols = logical(
            boundary.pack_input_tile(
                slot(row_stride=1, vector_layout="cols", block=[0, 0, 20, 1], valid=[0, 0, 20, 1], other=NEG_INF),
                host))
        self.assertEqual([cols[r * 32 + c] for r in (0, 31) for c in (0, 19)], [100, 119, 100, 119])
        self.assertEqual(cols[20], 0)

    def test_rejections(self):
        with self.assertRaisesRegex(ValueError, "outside its host buffer"):
            boundary.pack_input_tile(slot(), bf16_buffer(range(1000)))
        with self.assertRaisesRegex(ValueError, "unsupported cast"):
            boundary.pack_input_tile(slot(host_dtype="bf16", tile_dtype="f8E4M3FN"), bf16_buffer(range(1024)))
        with self.assertRaisesRegex(ValueError, "no host buffer for kernel argument 3"):
            boundary.pack_inputs(dict(version=2, inputs=[slot(arg=3)], outputs=[]), {0: bytes(2048)})
        with self.assertRaisesRegex(ValueError, "no host buffer for kernel argument 3"):
            boundary.unpack_outputs(dict(version=2, inputs=[], outputs=[slot(arg=3)]), bytes(2048), {})
        with self.assertRaisesRegex(ValueError, "version-2"):
            boundary.pack_inputs(dict(version=1, inputs=[]), {})

    def test_outputs_write_only_the_valid_rectangle(self):
        values = list(range(1024))
        data = pack_bf16_pair_halves(bf16_buffer(values))
        host = bytearray(b"\x55\x55" * (40 * 40))
        boundary.unpack_output_tile(slot(offset=4 * 40 + 9, row_stride=40, valid=[0, 0, 3, 7]), data, host)
        for row in range(40):
            for col in range(40):
                got = int.from_bytes(host[2 * (row * 40 + col):2 * (row * 40 + col) + 2], "little")
                inside = 4 <= row < 7 and 9 <= col < 16
                self.assertEqual(got, (row - 4) * 32 + (col - 9) if inside else 0x5555, (row, col))

    def test_outputs_widen_to_f32_and_read_vectors(self):
        data = pack_bf16_pair_halves(bf16_buffer([0x3F80] * 1024))
        host = bytearray(4 * 8)
        boundary.unpack_output_tile(slot(host_dtype="f32", row_stride=1, vector_layout="rows", valid=[0, 0, 5, 1]),
                                    data, host)
        self.assertEqual(struct.unpack("<8f", host), (1.0, ) * 5 + (0.0, ) * 3)

    def test_window_round_trip(self):
        manifest = dict(version=2,
                        inputs=[slot(index=0),
                                slot(index=1, arg=1, host_dtype="f8E4M3FN", tile_dtype="f8E4M3FN")], outputs=[])
        window = boundary.pack_inputs(manifest, {0: bf16_buffer(range(1024)), 1: bytes(range(256)) * 4})
        self.assertEqual(len(window), 2 * boundary.SLOT_BYTES)
        self.assertEqual(window[:2048], pack_bf16_pair_halves(bf16_buffer(range(1024))))
        self.assertEqual(window[2048:3072], bytes(range(256)) * 4)


if __name__ == "__main__":
    unittest.main()
