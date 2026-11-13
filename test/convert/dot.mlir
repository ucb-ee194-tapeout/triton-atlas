// RUN: triton-atlas-opt %s --convert-triton-to-atlas-virtual --verify-atlas-virtual-stream | FileCheck %s --implicit-check-not='tt.' --implicit-check-not='linalg.' --implicit-check-not='tensor.'

// Pass 3's linalg form of inputs/matmul_relu.ttir.mlir, with Pass 6 slots
// and Pass 4's MXU policy tag. Expected sequence matches the team's virtual
// reference: inputs, optional XLU weight orientation, MXU matmul, ReLU, output.
module {
  tt.func @matmul_relu(%x: !tt.ptr<f8E4M3FN>, %w: !tt.ptr<f8E4M3FN>, %out: !tt.ptr<bf16>) {
    %xp = tt.splat %x : !tt.ptr<f8E4M3FN> -> tensor<32x32x!tt.ptr<f8E4M3FN>>
    %wp = tt.splat %w : !tt.ptr<f8E4M3FN> -> tensor<32x32x!tt.ptr<f8E4M3FN>>
    %op = tt.splat %out : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    %a = tt.load %xp {ttatlas.slot = #ttatlas.slot<index = 0, arg = 0, role = activation_fp8, tile_origin = affine_map<()[s0, s1, s2, s3, s4, s5] -> (0, 0)>, logical_shape = [32, 32], host_dtype = f8E4M3FN>} : tensor<32x32x!tt.ptr<f8E4M3FN>>
    %b = tt.load %wp {ttatlas.slot = #ttatlas.slot<index = 1, arg = 1, role = weight_fp8_kn, tile_origin = affine_map<()[s0, s1, s2, s3, s4, s5] -> (0, 0)>, logical_shape = [32, 32], host_dtype = f8E4M3FN>} : tensor<32x32x!tt.ptr<f8E4M3FN>>
    %zero = arith.constant 0.0 : f32
    %empty = tensor.empty() : tensor<32x32xf32>
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<32x32xf32>) -> tensor<32x32xf32>
    %dot = linalg.matmul {ttatlas.mxu = 0 : i32} ins(%a, %b : tensor<32x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>) outs(%init : tensor<32x32xf32>) -> tensor<32x32xf32>
    %relu_empty = tensor.empty() : tensor<32x32xf32>
    %relu = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%dot : tensor<32x32xf32>) outs(%relu_empty : tensor<32x32xf32>) {
    ^bb0(%value: f32, %out_value: f32):
      %c0 = arith.constant 0.0 : f32
      %result = arith.maxnumf %value, %c0 : f32
      linalg.yield %result : f32
    } -> tensor<32x32xf32>
    %bf16 = arith.truncf %relu : tensor<32x32xf32> to tensor<32x32xbf16>
    tt.store %op, %bf16 {ttatlas.slot = #ttatlas.slot<index = 0, arg = 2, role = output_bf16, tile_origin = affine_map<()[s0, s1, s2, s3, s4, s5] -> (0, 0)>, logical_shape = [32, 32], host_dtype = bf16>} : tensor<32x32x!tt.ptr<bf16>>
    tt.return
  }

  tt.func @k_chain(%x: !tt.ptr<f8E4M3FN>, %w0: !tt.ptr<f8E4M3FN>, %w1: !tt.ptr<f8E4M3FN>, %out: !tt.ptr<bf16>) {
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
    %bf16 = arith.truncf %second : tensor<32x32xf32> to tensor<32x32xbf16>
    tt.store %op, %bf16 {ttatlas.slot = #ttatlas.slot<index = 0, arg = 3, role = output_bf16, tile_origin = affine_map<()[s0, s1, s2, s3, s4, s5] -> (0, 0)>, logical_shape = [32, 32], host_dtype = bf16>} : tensor<32x32x!tt.ptr<bf16>>
    tt.return
  }
}

// CHECK-LABEL: func.func @matmul_relu() -> !atlas.virtual_state
// CHECK: atlas.virtual_input_fp8
// CHECK: atlas.virtual_input_fp8
// CHECK: atlas.virtual_xlu_transpose
// CHECK: atlas.virtual_mxu_matmul
// CHECK-SAME: unit = 0 : i32
// CHECK: atlas.virtual_vpu_unary
// CHECK-SAME: kind = "relu"
// CHECK: atlas.virtual_output_bf16
// CHECK-LABEL: func.func @k_chain() -> !atlas.virtual_state
// CHECK: atlas.virtual_mxu_load_weight
// CHECK: atlas.virtual_mxu_reset
// CHECK: atlas.virtual_xlu_transpose
// CHECK: atlas.virtual_mxu_load_weight
// CHECK: atlas.virtual_mxu_accumulate
// CHECK: atlas.virtual_mxu_readout_bf16
