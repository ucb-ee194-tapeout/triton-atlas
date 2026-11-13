// RUN: not triton-atlas-opt %s --convert-triton-to-atlas-virtual 2>&1 | FileCheck %s

// A kernel whose compute pass 7 cannot lower yet (a division: the VPU ops pass 7 knows are add, sub, mul, min, max and
// relu) is rejected with the operation named, not converted halfway; the kernel is left as it was. copy.mlir is the same
// kernel without the compute, which converts; dot.mlir and elementwise.mlir cover compute that does lower.
// CHECK: error: atlas: unsupported linalg.generic: no Atlas VPU operation for arith.divf yet
module {
  tt.func @divide(%a: !tt.ptr<bf16>, %b: !tt.ptr<bf16>) {
    %p = tt.splat %a : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    %q = tt.splat %b : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    %v = tt.load %p {ttatlas.slot = #ttatlas.slot<index = 0, arg = 0, role = constant_bf16, tile_origin = affine_map<()[s0, s1, s2, s3, s4, s5] -> (0, 0)>, logical_shape = [32, 32], host_dtype = bf16>} : tensor<32x32x!tt.ptr<bf16>>
    %e = tensor.empty() : tensor<32x32xbf16>
    %s = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%v, %v : tensor<32x32xbf16>, tensor<32x32xbf16>) outs(%e : tensor<32x32xbf16>) {
    ^bb0(%x: bf16, %y: bf16, %o: bf16):
      %d = arith.divf %x, %y : bf16
      linalg.yield %d : bf16
    } -> tensor<32x32xbf16>
    tt.store %q, %s {ttatlas.slot = #ttatlas.slot<index = 0, arg = 1, role = output_bf16, tile_origin = affine_map<()[s0, s1, s2, s3, s4, s5] -> (0, 0)>, logical_shape = [32, 32], host_dtype = bf16>} : tensor<32x32x!tt.ptr<bf16>>
    tt.return
  }
}
