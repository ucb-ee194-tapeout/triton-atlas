// Test input: the probe kernel strided_cols after passes 1-3 (linalg compute, TTIR memory ops), as pass 5 and pass 4 receive it.
module {
  tt.func public @strided_cols(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>) attributes {noinline = false} {
    %cst = arith.constant dense<32> : tensor<32x1xi32>
    %cst_0 = arith.constant dense<2> : tensor<1x32xi32>
    %cst_1 = arith.constant dense<64> : tensor<32x1xi32>
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %2 = arith.muli %1, %cst_1 : tensor<32x1xi32>
    %3 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>>
    %4 = tt.addptr %3, %2 : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
    %5 = tt.expand_dims %0 {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %6 = arith.muli %5, %cst_0 : tensor<1x32xi32>
    %7 = tt.broadcast %4 : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x32x!tt.ptr<bf16>>
    %8 = tt.broadcast %6 : tensor<1x32xi32> -> tensor<32x32xi32>
    %9 = tt.addptr %7, %8 : tensor<32x32x!tt.ptr<bf16>>, tensor<32x32xi32>
    %10 = tt.load %9 : tensor<32x32x!tt.ptr<bf16>>
    %11 = arith.muli %1, %cst : tensor<32x1xi32>
    %12 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<32x1x!tt.ptr<bf16>>
    %13 = tt.addptr %12, %11 : tensor<32x1x!tt.ptr<bf16>>, tensor<32x1xi32>
    %14 = tt.broadcast %13 : tensor<32x1x!tt.ptr<bf16>> -> tensor<32x32x!tt.ptr<bf16>>
    %15 = tt.broadcast %5 : tensor<1x32xi32> -> tensor<32x32xi32>
    %16 = tt.addptr %14, %15 : tensor<32x32x!tt.ptr<bf16>>, tensor<32x32xi32>
    tt.store %16, %10 : tensor<32x32x!tt.ptr<bf16>>
    tt.return
  }
}
