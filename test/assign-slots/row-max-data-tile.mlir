// RUN: triton-atlas-opt %s --ttatlas-assign-slots | FileCheck %s

// A row max over a load whose padded columns the host fills with -inf: the tile is data. Its slot keeps a data role (the
// -inf is its fill, in the pad tag), not a mask role: only a ttatlas.mask_apply constant makes a mask slot.
// CHECK: tt.load {{.*}}role = constant_bf16
// CHECK-NOT: mask_bf16
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  tt.func public @k3(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>) {
    %cst = arith.constant 0xFF80 : bf16
    %cst_0 = arith.constant dense<0xFF80> : tensor<32x32xbf16>
    %cst_1 = arith.constant dense<20> : tensor<1x32xi32>
    %cst_2 = arith.constant dense<32> : tensor<32x1xi32>
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %2 = tt.expand_dims %0 {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %3 = arith.muli %1, %cst_2 : tensor<32x1xi32>
    %4 = tt.broadcast %3 : tensor<32x1xi32> -> tensor<32x32xi32>
    %5 = tt.broadcast %2 : tensor<1x32xi32> -> tensor<32x32xi32>
    %6 = arith.addi %4, %5 : tensor<32x32xi32>
    %7 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    %8 = tt.addptr %7, %6 : tensor<32x32x!tt.ptr<bf16>>, tensor<32x32xi32>
    %9 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    %10 = tt.addptr %9, %6 : tensor<32x32x!tt.ptr<bf16>>, tensor<32x32xi32>
    %11 = arith.cmpi slt, %2, %cst_1 : tensor<1x32xi32>
    %12 = tt.broadcast %11 : tensor<1x32xi1> -> tensor<32x32xi1>
    %13 = tt.load %8 {ttatlas.host_dtype = bf16, ttatlas.pad = #ttatlas.pad<neutral = neg_inf, logical_shape = [32, 20]>} : tensor<32x32x!tt.ptr<bf16>>
    %14 = tensor.empty() : tensor<32xbf16>
    %15 = linalg.fill ins(%cst : bf16) outs(%14 : tensor<32xbf16>) -> tensor<32xbf16>
    %reduced = linalg.reduce { arith.maximumf } ins(%13 : tensor<32x32xbf16>) outs(%15 : tensor<32xbf16>) dimensions = [1]
    %16 = tensor.empty() : tensor<32x32xbf16>
    %broadcasted = linalg.broadcast ins(%reduced : tensor<32xbf16>) outs(%16 : tensor<32x32xbf16>) dimensions = [1]
    %17 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%broadcasted, %broadcasted : tensor<32x32xbf16>, tensor<32x32xbf16>) outs(%16 : tensor<32x32xbf16>) {
    ^bb0(%in: bf16, %in_3: bf16, %out: bf16):
      %18 = arith.addf %in, %in_3 : bf16
      linalg.yield %18 : bf16
    } -> tensor<32x32xbf16>
    tt.store %10, %17 {ttatlas.host_dtype = bf16} : tensor<32x32x!tt.ptr<bf16>>
    tt.return
  }
}
