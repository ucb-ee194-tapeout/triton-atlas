// RUN: triton-atlas-opt %s --convert-triton-to-atlas-virtual --verify-atlas-virtual-stream | FileCheck %s --implicit-check-not='tt.' --implicit-check-not='linalg.' --implicit-check-not='tensor.'

// The forms pass 5 leaves after a matmul chain: a linalg.generic{arith.truncf f32 -> bf16} as the readout marker, then a
// ReLU written as a max with a zero tile (a tensor constant read only by the max). The marker becomes the chain's
// mxu_readout_bf16, the ReLU a VPU relu, and the constant is folded away.
module {
  tt.func @k_chain_marker(%x: !tt.ptr<f8E4M3FN>, %w0: !tt.ptr<f8E4M3FN>, %w1: !tt.ptr<f8E4M3FN>, %out: !tt.ptr<bf16>) {
    %xp = tt.splat %x : !tt.ptr<f8E4M3FN> -> tensor<32x32x!tt.ptr<f8E4M3FN>>
    %w0p = tt.splat %w0 : !tt.ptr<f8E4M3FN> -> tensor<32x32x!tt.ptr<f8E4M3FN>>
    %w1p = tt.splat %w1 : !tt.ptr<f8E4M3FN> -> tensor<32x32x!tt.ptr<f8E4M3FN>>
    %op = tt.splat %out : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    %a = tt.load %xp {ttatlas.slot = #ttatlas.slot<index = 0, arg = 0, role = activation_fp8, tile_origin = affine_map<()[s0, s1, s2, s3, s4, s5] -> (0, 0)>, logical_shape = [32, 32], host_dtype = f8E4M3FN>} : tensor<32x32x!tt.ptr<f8E4M3FN>>
    %b0 = tt.load %w0p {ttatlas.slot = #ttatlas.slot<index = 1, arg = 1, role = weight_fp8_kn, tile_origin = affine_map<()[s0, s1, s2, s3, s4, s5] -> (0, 0)>, logical_shape = [32, 32], host_dtype = f8E4M3FN>} : tensor<32x32x!tt.ptr<f8E4M3FN>>
    %b1 = tt.load %w1p {ttatlas.slot = #ttatlas.slot<index = 2, arg = 2, role = weight_fp8_kn, tile_origin = affine_map<()[s0, s1, s2, s3, s4, s5] -> (0, 0)>, logical_shape = [32, 32], host_dtype = f8E4M3FN>} : tensor<32x32x!tt.ptr<f8E4M3FN>>
    %zero = arith.constant 0.0 : f32
    %empty = tensor.empty() : tensor<32x32xf32>
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<32x32xf32>) -> tensor<32x32xf32>
    %first = linalg.matmul {ttatlas.mxu = 0 : i32} ins(%a, %b0 : tensor<32x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>) outs(%init : tensor<32x32xf32>) -> tensor<32x32xf32>
    %second = linalg.matmul {ttatlas.mxu = 0 : i32} ins(%a, %b1 : tensor<32x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>) outs(%first : tensor<32x32xf32>) -> tensor<32x32xf32>
    %out_empty = tensor.empty() : tensor<32x32xbf16>
    %bf16 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%second : tensor<32x32xf32>) outs(%out_empty : tensor<32x32xbf16>) {
    ^bb0(%in: f32, %out_value: bf16):
      %narrow = arith.truncf %in : f32 to bf16
      linalg.yield %narrow : bf16
    } -> tensor<32x32xbf16>
    %zero_tile = arith.constant dense<0.000000e+00> : tensor<32x32xbf16>
    %relu_empty = tensor.empty() : tensor<32x32xbf16>
    %relu = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%bf16, %zero_tile : tensor<32x32xbf16>, tensor<32x32xbf16>) outs(%relu_empty : tensor<32x32xbf16>) {
    ^bb0(%in: bf16, %zero_value: bf16, %out_value: bf16):
      %max = arith.maxnumf %in, %zero_value : bf16
      linalg.yield %max : bf16
    } -> tensor<32x32xbf16>
    tt.store %op, %relu {ttatlas.slot = #ttatlas.slot<index = 0, arg = 3, role = output_bf16, tile_origin = affine_map<()[s0, s1, s2, s3, s4, s5] -> (0, 0)>, logical_shape = [32, 32], host_dtype = bf16>} : tensor<32x32x!tt.ptr<bf16>>
    tt.return
  }
}

// CHECK-LABEL: func.func @k_chain_marker() -> !atlas.virtual_state
// CHECK: atlas.virtual_mxu_reset
// CHECK: atlas.virtual_mxu_accumulate
// CHECK: atlas.virtual_mxu_readout_bf16
// CHECK: atlas.virtual_vpu_unary
// CHECK-SAME: kind = "relu"
// CHECK: atlas.virtual_output_bf16
