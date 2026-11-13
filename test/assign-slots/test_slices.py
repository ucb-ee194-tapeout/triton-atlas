"""Adversarial slice contracts and Linalg operand orientation."""
from pathlib import Path
import shutil
import subprocess
import unittest

SOURCE = (Path(__file__).parents[1] / "inputs/boundary-slices.ttir.mlir").read_text()


class SliceContractTest(unittest.TestCase):

    def run_pass(self, source, *passes):
        return subprocess.run([shutil.which("triton-atlas-opt"), *passes], input=source, capture_output=True, text=True)

    def test_unsupported_slice_forms(self):
        cases = [
            (SOURCE.replace('%b: !tt.ptr<bf16>',
                            '%b: !tt.ptr<bf16>, %dynamic: index').replace('%x[0, 32]', '%x[0, %dynamic]'),
             'constant 32x32 extract_slice'),
            (SOURCE.replace('dense<64>', 'dense<65>'), 'contiguous row-major tile'),
            (SOURCE.replace('%x[0, 32]', '%x[0, 64]'), 'out-of-bounds'),
            (SOURCE.replace('%o0[0, 32]', '%o0[0, 0]'), 'disjoint'),
            (SOURCE.replace('tt.store %q, %o1', 'tt.store %q, %o0'), 'unshared'),
            (SOURCE.replace(
                '%left = tensor.extract_slice',
                '%escape = tensor.cast %x : tensor<32x64xbf16> to tensor<?x?xbf16>\n    %left = tensor.extract_slice'),
             'only constant 32x32 extract_slice'),
        ]
        for source, diagnostic in cases:
            for pipeline in [('--ttatlas-assign-slots', ), ('--convert-triton-to-atlas-virtual', )]:
                with self.subTest(diagnostic=diagnostic, pipeline=pipeline):
                    result = self.run_pass(source, *pipeline)
                    self.assertNotEqual(result.returncode, 0)
                    # Boundary conversion consumes slots rather than re-proving
                    # pointers, so only Pass 6 diagnoses a noncontiguous pointer.
                    if diagnostic != 'contiguous row-major tile' or pipeline[0] == '--ttatlas-assign-slots':
                        self.assertIn(diagnostic, result.stderr)

    def test_repeated_output_tile_is_rejected(self):
        source = (Path(__file__).parent / "bf16.mlir").read_text().replace(
            "tt.return", "tt.store %p, %x {ttatlas.host_dtype = bf16} : tensor<32x32x!tt.ptr<bf16>>\n"
            "tt.store %p, %x {ttatlas.host_dtype = bf16} : tensor<32x32x!tt.ptr<bf16>>\n tt.return")
        result = self.run_pass(source, "--ttatlas-assign-slots")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("repeated writes to one logical output tile", result.stderr)

    def test_parent_logical_shape_is_sliced(self):
        source = SOURCE.replace(
            "ttatlas.host_dtype = bf16", "ttatlas.host_dtype = bf16, "
            "ttatlas.pad = #ttatlas.pad<neutral = zero, logical_shape = [32, 64]>")
        result = self.run_pass(source, "--ttatlas-assign-slots", "--ttatlas-verify-contract=stage=6")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("logical_shape = [32, 32], host_dtype"), 5)

    def test_parent_partial_shape_is_intersected(self):
        source = SOURCE.replace(
            "ttatlas.host_dtype = bf16", "ttatlas.host_dtype = bf16, "
            "ttatlas.pad = #ttatlas.pad<neutral = zero, logical_shape = [17, 47]>")
        source = source.replace(
            "ttatlas.host_dtype = f32", "ttatlas.host_dtype = f32, "
            "ttatlas.pad = #ttatlas.pad<neutral = zero, logical_shape = [17, 47]>")
        result = self.run_pass(source, "--ttatlas-assign-slots", "--ttatlas-verify-contract=stage=6")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("logical_shape = [17, 32], host_dtype"), 3)
        self.assertEqual(result.stdout.count("logical_shape = [17, 15], host_dtype"), 2)

    def test_invalid_pad_extent_is_rejected(self):
        source = SOURCE.replace(
            "ttatlas.host_dtype = bf16", "ttatlas.host_dtype = bf16, "
            "ttatlas.pad = #ttatlas.pad<neutral = zero, logical_shape = [33, 64]>")
        result = self.run_pass(source, "--ttatlas-assign-slots")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("outside the boundary extent", result.stderr)

    def test_stage6_requires_slice_slots(self):
        result = self.run_pass(SOURCE, '--ttatlas-verify-contract=stage=6')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stderr.count('error: contract of pass 6 broken: missing #ttatlas.slot'), 5)

    def test_linalg_fp8_orientation_and_reuse(self):
        source = SOURCE[:SOURCE.index('    %out =')].replace('bf16', 'f8E4M3FN')
        # The B tile is transposed explicitly, so its host orientation is NK.
        source += '''    %empty = tensor.empty() : tensor<32x32xf8E4M3FN>
    %bt = linalg.transpose ins(%right : tensor<32x32xf8E4M3FN>) outs(%empty : tensor<32x32xf8E4M3FN>) permutation = [1, 0]
    %zero = arith.constant 0.0 : f32
    %acc = tensor.empty() : tensor<32x32xf32>
    %fill = linalg.fill ins(%zero : f32) outs(%acc : tensor<32x32xf32>) -> tensor<32x32xf32>
    %dot = linalg.matmul ins(%left, %bt : tensor<32x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>) outs(%fill : tensor<32x32xf32>) -> tensor<32x32xf32>
    %dot2 = linalg.matmul ins(%again, %bt : tensor<32x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>) outs(%dot : tensor<32x32xf32>) -> tensor<32x32xf32>
    tt.return
  }
}
'''
        result = self.run_pass(source, '--ttatlas-assign-slots', '--ttatlas-verify-contract=stage=6')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count('index = 0, arg = 0, role = activation_fp8'), 2)
        self.assertIn('index = 1, arg = 0, role = weight_fp8_nk', result.stdout)


if __name__ == '__main__':
    unittest.main()
