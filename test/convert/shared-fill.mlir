// RUN: triton-atlas-opt %s --convert-triton-to-atlas-virtual --verify-atlas-virtual-stream | FileCheck %s --implicit-check-not='tt.' --implicit-check-not='linalg.' --implicit-check-not='tensor.'

// Two independent matmuls seeded by one linalg.fill, as CSE leaves them: the fill is shared, so lowering the first matmul
// must not take it away from the second.
module {
  tt.func @shared_fill(%x: !tt.ptr<f8E4M3FN>, %w0: !tt.ptr<f8E4M3FN>, %w1: !tt.ptr<f8E4M3FN>, %o0: !tt.ptr<bf16>, %o1: !tt.ptr<bf16>) {
    %xp = tt.splat %x : !tt.ptr<f8E4M3FN> -> tensor<32x32x!tt.ptr<f8E4M3FN>>
    %w0p = tt.splat %w0 : !tt.ptr<f8E4M3FN> -> tensor<32x32x!tt.ptr<f8E4M3FN>>
    %w1p = tt.splat %w1 : !tt.ptr<f8E4M3FN> -> tensor<32x32x!tt.ptr<f8E4M3FN>>
    %o0p = tt.splat %o0 : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    %o1p = tt.splat %o1 : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    %a = tt.load %xp {ttatlas.slot = #ttatlas.slot<index = 0, arg = 0, role = activation_fp8, tile_origin = affine_map<()[s0, s1, s2, s3, s4, s5] -> (0, 0)>, logical_shape = [32, 32], host_dtype = f8E4M3FN>} : tensor<32x32x!tt.ptr<f8E4M3FN>>
    %b0 = tt.load %w0p {ttatlas.slot = #ttatlas.slot<index = 1, arg = 1, role = weight_fp8_kn, tile_origin = affine_map<()[s0, s1, s2, s3, s4, s5] -> (0, 0)>, logical_shape = [32, 32], host_dtype = f8E4M3FN>} : tensor<32x32x!tt.ptr<f8E4M3FN>>
    %b1 = tt.load %w1p {ttatlas.slot = #ttatlas.slot<index = 2, arg = 2, role = weight_fp8_kn, tile_origin = affine_map<()[s0, s1, s2, s3, s4, s5] -> (0, 0)>, logical_shape = [32, 32], host_dtype = f8E4M3FN>} : tensor<32x32x!tt.ptr<f8E4M3FN>>
    %zero = arith.constant 0.0 : f32
    %empty = tensor.empty() : tensor<32x32xf32>
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<32x32xf32>) -> tensor<32x32xf32>
    %m0 = linalg.matmul {ttatlas.mxu = 0 : i32} ins(%a, %b0 : tensor<32x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>) outs(%init : tensor<32x32xf32>) -> tensor<32x32xf32>
    %m1 = linalg.matmul {ttatlas.mxu = 0 : i32} ins(%a, %b1 : tensor<32x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>) outs(%init : tensor<32x32xf32>) -> tensor<32x32xf32>
    %e0 = tensor.empty() : tensor<32x32xbf16>
    %r0 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%m0 : tensor<32x32xf32>) outs(%e0 : tensor<32x32xbf16>) {
    ^bb0(%in: f32, %out_value: bf16):
      %narrow = arith.truncf %in : f32 to bf16
      linalg.yield %narrow : bf16
    } -> tensor<32x32xbf16>
    %e1 = tensor.empty() : tensor<32x32xbf16>
    %r1 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%m1 : tensor<32x32xf32>) outs(%e1 : tensor<32x32xbf16>) {
    ^bb0(%in: f32, %out_value: bf16):
      %narrow = arith.truncf %in : f32 to bf16
      linalg.yield %narrow : bf16
    } -> tensor<32x32xbf16>
    tt.store %o0p, %r0 {ttatlas.slot = #ttatlas.slot<index = 0, arg = 3, role = output_bf16, tile_origin = affine_map<()[s0, s1, s2, s3, s4, s5] -> (0, 0)>, logical_shape = [32, 32], host_dtype = bf16>} : tensor<32x32x!tt.ptr<bf16>>
    tt.store %o1p, %r1 {ttatlas.slot = #ttatlas.slot<index = 1, arg = 4, role = output_bf16, tile_origin = affine_map<()[s0, s1, s2, s3, s4, s5] -> (0, 0)>, logical_shape = [32, 32], host_dtype = bf16>} : tensor<32x32x!tt.ptr<bf16>>
    tt.return
  }
}

// CHECK-LABEL: func.func @shared_fill() -> !atlas.virtual_state
// CHECK-COUNT-2: atlas.virtual_mxu_matmul
// CHECK-COUNT-2: atlas.virtual_output_bf16
