"""Physical byte packing/cropping, independent of metadata validity."""
import unittest
import json
import re
import shutil
import subprocess
from pathlib import Path
from triton_atlas_backend.boundary import pack_input_tile, unpack_output_tile
from jsonschema import Draft202012Validator


def slot(shape=(32, 32), role="constant_bf16", neutral=None, origin=(0, 0)):
    result = dict(host_dtype="bf16", logical_shape=list(shape), role=role,
                  tile_origin=f"affine_map<()[s0, s1, s2, s3, s4, s5] -> ({origin[0]}, {origin[1]})>")
    if neutral:
        result["pad_neutral"] = neutral
    return result


class BufferTest(unittest.TestCase):

    def compile(self, source):
        result = subprocess.run([
            shutil.which("triton-atlas-opt"), "--ttatlas-assign-slots", "--ttatlas-verify-contract=stage=6",
            "--convert-triton-to-atlas-virtual", "--ttatlas-verify-contract=stage=7", "--verify-atlas-virtual-stream"
        ], input=source, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        text = re.search(r'ttatlas.kernel = "((?:[^"\\]|\\.)*)"', result.stdout).group(1)
        manifest = json.loads(text.replace(r'\22', '"').replace(r'\5C', '\\'))
        schema = Path(__file__).parents[2] / "python/triton_atlas_backend/schemas/ttatlas.kernel.v1.schema.json"
        Draft202012Validator(json.loads(schema.read_text())).validate(manifest)
        return manifest

    def test_native_partial_input_and_output(self):
        source = (Path(__file__).parents[1] / "assign-slots/pad-partial.mlir").read_text()
        source = source.replace(
            "tt.return", "tt.store %p, %x {ttatlas.host_dtype = bf16} : "
            "tensor<32x32x!tt.ptr<bf16>>\n    tt.return")
        metadata = self.compile(source)
        self.assertEqual(metadata["inputs"][0]["logical_shape"], [16, 32])
        self.assertEqual(metadata["inputs"][0]["pad_neutral"], "zero")
        self.assertEqual(metadata["outputs"][0]["logical_shape"], [16, 32])
        negative = self.compile(source.replace("neutral = zero", "neutral = neg_inf"))
        self.assertEqual(negative["inputs"][0]["pad_neutral"], "neg_inf")
        self.assertEqual(negative["inputs"][0]["role"], "constant_bf16")  # a pad value never sets a mask role

    def test_native_bias_slot(self):
        root = Path(__file__).parents[1] / "assign-slots"
        bias = (root / "bias-1d.mlir").read_text().replace("%a: !tt.ptr<bf16>", "%a: !tt.ptr<bf16>, %b: !tt.ptr<bf16>")
        copy = (root / "bf16.mlir").read_text()
        body = copy[copy.index("    %r ="):copy.index("    tt.return")]
        body = body.replace("%", "%copy_").replace("%copy_a", "%b")
        bias = bias.replace(
            "  tt.return", body + "    tt.store %copy_p, %copy_x {ttatlas.host_dtype = bf16} : "
            "tensor<32x32x!tt.ptr<bf16>>\n  tt.return")
        bias = bias.replace("start = 0 : i32, end = 32 : i32", "start = 32 : i32, end = 64 : i32", 1)
        metadata = self.compile(bias)
        self.assertTrue(metadata["inputs"][0]["tile_origin"].endswith("(0, 32)>"))
        self.assertEqual(metadata["inputs"][0]["role"], "bias_bf16")
        self.assertEqual(metadata["inputs"][0]["logical_shape"], [1, 32])
        partial = bias.replace(
            "ttatlas.host_dtype = bf16}", "ttatlas.host_dtype = bf16, ttatlas.pad = "
            "#ttatlas.pad<neutral = zero, logical_shape = [1, 7]>}", 1)
        metadata = self.compile(partial)
        self.assertEqual(metadata["inputs"][0]["logical_shape"], [1, 7])

    def test_full_tile(self):
        host = bytes(range(256)) * 8
        self.assertEqual(pack_input_tile(host, (32, 32), slot()), host)

    def test_fp8_zero_padding(self):
        entry = slot((2, 3), neutral="zero")
        entry["host_dtype"] = "f8E4M3FN"
        self.assertEqual(pack_input_tile(bytes([56]) * 6, (2, 3), entry),
                         (bytes([56]) * 3 + bytes(29)) * 2 + bytes(30 * 32))

    def test_neutral_and_offsets(self):
        host = b"\x80\x3f" * (20 * 40)
        for neutral, fill in [("zero", b"\0\0"), ("neg_inf", b"\x80\xff"), ("one", b"\x80\x3f"),
                              ("pos_inf", b"\x80\x7f")]:
            packed = pack_input_tile(host, (20, 40), slot((3, 7), neutral=neutral, origin=(4, 9)))
            for r in range(32):
                for c in range(32):
                    self.assertEqual(packed[2 * (r * 32 + c):2 * (r * 32 + c + 1)],
                                     b"\x80\x3f" if r < 3 and c < 7 else fill)

    def test_cropping_preserves_surrounding_bytes(self):
        host = bytearray(b"\x55\x55" * (20 * 40))
        unpack_output_tile(b"\x80\x3f" * 1024, host, (20, 40), slot((3, 7), role="output_bf16", origin=(4, 9)))
        for r in range(20):
            for c in range(40):
                self.assertEqual(host[2 * (r * 40 + c):2 * (r * 40 + c + 1)],
                                 b"\x80\x3f" if 4 <= r < 7 and 9 <= c < 16 else b"\x55\x55")

    def test_bias_broadcast(self):
        packed = pack_input_tile(b"\x80\x3f" * 35, (35, ), slot((1, 3), role="bias_bf16", origin=(0, 32)))
        self.assertEqual(packed, (b"\x80\x3f" * 3 + bytes(58)) * 32)

    def test_f32_output_widens_bf16_and_preserves_surroundings(self):
        entry = slot((1, 4), role="output_bf16", origin=(1, 2))
        entry["host_dtype"] = "f32"
        host = bytearray(b"\x55" * (3 * 8 * 4))
        # +1, -2, +0, -0: widening inserts low zero bits, with no extra rounding.
        tile = bytes.fromhex("803f00c000000080") + bytes(2040)
        unpack_output_tile(tile, host, (3, 8), entry)
        expected = bytearray(b"\x55" * len(host))
        expected[40:56] = bytes.fromhex("0000803f000000c00000000000000080")
        self.assertEqual(host, expected)
        with self.assertRaisesRegex(ValueError, "buffer length"):
            unpack_output_tile(tile, bytearray(3 * 8 * 2), (3, 8), entry)

    def test_missing_neutral_and_fp8_infinity_rejected(self):
        with self.assertRaisesRegex(ValueError, "requires Pass 4"):
            pack_input_tile(bytes(8), (2, 2), slot((2, 2)))
        for neutral in ("neg_inf", "pos_inf"):
            entry = slot((2, 2), neutral=neutral)
            entry["host_dtype"] = "f8E4M3FN"
            with self.assertRaisesRegex(ValueError, "no infinity"):
                pack_input_tile(bytes(4), (2, 2), entry)
        # 1.0 does exist in E4M3FN (0x38).
        one = slot((1, 1), neutral="one")
        one["host_dtype"] = "f8E4M3FN"
        self.assertEqual(pack_input_tile(bytes([56]), (1, 1), one), bytes([56]) * 1024)


if __name__ == "__main__":
    unittest.main()
