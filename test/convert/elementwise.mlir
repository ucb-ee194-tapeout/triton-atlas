// RUN: triton-atlas-opt %s --convert-triton-to-atlas-virtual --verify-atlas-virtual-stream | FileCheck %s --implicit-check-not='tt.' --implicit-check-not='linalg.' --implicit-check-not='tensor.'

module {
  tt.func @elementwise(%x: !tt.ptr<bf16>, %y: !tt.ptr<bf16>, %out: !tt.ptr<bf16>) {
    %xp = tt.splat %x : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    %yp = tt.splat %y : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    %op = tt.splat %out : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    %a = tt.load %xp {ttatlas.slot = #ttatlas.slot<index = 0, arg = 0, role = bias_bf16, tile_origin = affine_map<()[s0, s1, s2, s3, s4, s5] -> (0, 0)>, logical_shape = [32, 32], host_dtype = bf16>} : tensor<32x32x!tt.ptr<bf16>>
    %b = tt.load %yp {ttatlas.slot = #ttatlas.slot<index = 1, arg = 1, role = bias_bf16, tile_origin = affine_map<()[s0, s1, s2, s3, s4, s5] -> (0, 0)>, logical_shape = [32, 32], host_dtype = bf16>} : tensor<32x32x!tt.ptr<bf16>>
    %empty0 = tensor.empty() : tensor<32x32xbf16>
    %add = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%a, %b : tensor<32x32xbf16>, tensor<32x32xbf16>) outs(%empty0 : tensor<32x32xbf16>) {
    ^bb0(%lhs: bf16, %rhs: bf16, %out_value: bf16):
      %value = arith.addf %lhs, %rhs : bf16
      linalg.yield %value : bf16
    } -> tensor<32x32xbf16>
    %empty1 = tensor.empty() : tensor<32x32xbf16>
    %sub = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%a, %b : tensor<32x32xbf16>, tensor<32x32xbf16>) outs(%empty1 : tensor<32x32xbf16>) {
    ^bb0(%lhs: bf16, %rhs: bf16, %out_value: bf16):
      %value = arith.subf %lhs, %rhs : bf16
      linalg.yield %value : bf16
    } -> tensor<32x32xbf16>
    %empty2 = tensor.empty() : tensor<32x32xbf16>
    %mul = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%a, %b : tensor<32x32xbf16>, tensor<32x32xbf16>) outs(%empty2 : tensor<32x32xbf16>) {
    ^bb0(%lhs: bf16, %rhs: bf16, %out_value: bf16):
      %value = arith.mulf %lhs, %rhs : bf16
      linalg.yield %value : bf16
    } -> tensor<32x32xbf16>
    %empty3 = tensor.empty() : tensor<32x32xbf16>
    %min = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%a, %b : tensor<32x32xbf16>, tensor<32x32xbf16>) outs(%empty3 : tensor<32x32xbf16>) {
    ^bb0(%lhs: bf16, %rhs: bf16, %out_value: bf16):
      %value = arith.minnumf %lhs, %rhs : bf16
      linalg.yield %value : bf16
    } -> tensor<32x32xbf16>
    %empty4 = tensor.empty() : tensor<32x32xbf16>
    %max = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%a, %b : tensor<32x32xbf16>, tensor<32x32xbf16>) outs(%empty4 : tensor<32x32xbf16>) {
    ^bb0(%lhs: bf16, %rhs: bf16, %out_value: bf16):
      %value = arith.maxnumf %lhs, %rhs : bf16
      linalg.yield %value : bf16
    } -> tensor<32x32xbf16>
    %empty5 = tensor.empty() : tensor<32x32xbf16>
    %relu = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%max : tensor<32x32xbf16>) outs(%empty5 : tensor<32x32xbf16>) {
    ^bb0(%value: bf16, %out_value: bf16):
      %zero = arith.constant 0.0 : bf16
      %result = arith.maxnumf %value, %zero : bf16
      linalg.yield %result : bf16
    } -> tensor<32x32xbf16>
    tt.store %op, %relu {ttatlas.slot = #ttatlas.slot<index = 0, arg = 2, role = output_bf16, tile_origin = affine_map<()[s0, s1, s2, s3, s4, s5] -> (0, 0)>, logical_shape = [32, 32], host_dtype = bf16>} : tensor<32x32x!tt.ptr<bf16>>
    tt.return
  }
}

// CHECK-LABEL: func.func @elementwise() -> !atlas.virtual_state
// CHECK: atlas.virtual_vpu_binary {{.*}}kind = "add"
// CHECK: atlas.virtual_vpu_binary {{.*}}kind = "sub"
// CHECK: atlas.virtual_vpu_binary {{.*}}kind = "mul"
// CHECK: atlas.virtual_vpu_binary {{.*}}kind = "min"
// CHECK: atlas.virtual_vpu_binary {{.*}}kind = "max"
// CHECK: atlas.virtual_vpu_unary {{.*}}kind = "relu"
// CHECK: atlas.virtual_output_bf16
