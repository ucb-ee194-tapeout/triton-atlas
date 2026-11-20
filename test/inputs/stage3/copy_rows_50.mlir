// Test input: the probe kernel copy_rows_50 after passes 1-3 (linalg compute, TTIR memory ops), as pass 5 and pass 4 receive it.
module {
  tt.func public @copy_rows_50(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x32xbf16>
    %cst_0 = arith.constant dense<50> : tensor<64xi32>
    %cst_1 = arith.constant dense<32> : tensor<64x1xi32>
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %1 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %2 = tt.expand_dims %0 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %3 = arith.muli %2, %cst_1 : tensor<64x1xi32>
    %4 = tt.expand_dims %1 {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %5 = tt.broadcast %3 : tensor<64x1xi32> -> tensor<64x32xi32>
    %6 = tt.broadcast %4 : tensor<1x32xi32> -> tensor<64x32xi32>
    %7 = arith.addi %5, %6 : tensor<64x32xi32>
    %8 = arith.cmpi slt, %0, %cst_0 : tensor<64xi32>
    %9 = tt.expand_dims %8 {axis = 1 : i32} : tensor<64xi1> -> tensor<64x1xi1>
    %10 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<64x32x!tt.ptr<bf16>>
    %11 = tt.addptr %10, %7 : tensor<64x32x!tt.ptr<bf16>>, tensor<64x32xi32>
    %12 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<64x32x!tt.ptr<bf16>>
    %13 = tt.addptr %12, %7 : tensor<64x32x!tt.ptr<bf16>>, tensor<64x32xi32>
    %14 = tt.broadcast %9 : tensor<64x1xi1> -> tensor<64x32xi1>
    %15 = tt.load %13, %14, %cst : tensor<64x32x!tt.ptr<bf16>>
    tt.store %11, %15, %14 : tensor<64x32x!tt.ptr<bf16>>
    tt.return
  }
}
