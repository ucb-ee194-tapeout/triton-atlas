// RUN: not triton-atlas-opt %s --convert-triton-to-atlas-virtual 2>&1 | FileCheck %s

// A reduction pass 7 must not lower: a start value that is not the combiner's identity (the virtual row reduction takes
// no start value, so it would be dropped).
// CHECK: atlas: unsupported linalg.reduce: the reduction must start from a linalg.fill of its combiner's identity
module {
  tt.func @bad_init(%x: !tt.ptr<bf16>, %out: !tt.ptr<bf16>) {
    %xp = tt.splat %x : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    %op = tt.splat %out : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    %input = tt.load %xp {ttatlas.slot = #ttatlas.slot<index = 0, arg = 0, role = bias_bf16, tile_origin = affine_map<()[s0, s1, s2, s3, s4, s5] -> (0, 0)>, logical_shape = [32, 32], host_dtype = bf16>} : tensor<32x32x!tt.ptr<bf16>>
    %start = arith.constant 1.000000e+00 : bf16
    %row_empty = tensor.empty() : tensor<32xbf16>
    %row_init = linalg.fill ins(%start : bf16) outs(%row_empty : tensor<32xbf16>) -> tensor<32xbf16>
    %row = linalg.reduce ins(%input : tensor<32x32xbf16>) outs(%row_init : tensor<32xbf16>) dimensions = [1]
    (%lhs: bf16, %rhs: bf16) {
      %r = arith.addf %lhs, %rhs : bf16
      linalg.yield %r : bf16
    }
    %tile_empty = tensor.empty() : tensor<32x32xbf16>
    %tile = linalg.broadcast ins(%row : tensor<32xbf16>) outs(%tile_empty : tensor<32x32xbf16>) dimensions = [1]
    tt.store %op, %tile {ttatlas.slot = #ttatlas.slot<index = 0, arg = 1, role = output_bf16, tile_origin = affine_map<()[s0, s1, s2, s3, s4, s5] -> (0, 0)>, logical_shape = [32, 32], host_dtype = bf16>} : tensor<32x32x!tt.ptr<bf16>>
    tt.return
  }
}
