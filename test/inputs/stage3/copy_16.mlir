// Test input: the probe kernel copy_16 after passes 1-3 (linalg compute, TTIR memory ops), as pass 5 and pass 4 receive it.
module {
  tt.func public @copy_16(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>) attributes {noinline = false} {
    %cst = arith.constant dense<16> : tensor<16x1xi32>
    %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %2 = arith.muli %1, %cst : tensor<16x1xi32>
    %3 = tt.expand_dims %0 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %4 = tt.broadcast %2 : tensor<16x1xi32> -> tensor<16x16xi32>
    %5 = tt.broadcast %3 : tensor<1x16xi32> -> tensor<16x16xi32>
    %6 = arith.addi %4, %5 : tensor<16x16xi32>
    %7 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<16x16x!tt.ptr<bf16>>
    %8 = tt.addptr %7, %6 : tensor<16x16x!tt.ptr<bf16>>, tensor<16x16xi32>
    %9 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<16x16x!tt.ptr<bf16>>
    %10 = tt.addptr %9, %6 : tensor<16x16x!tt.ptr<bf16>>, tensor<16x16xi32>
    %11 = tt.load %10 : tensor<16x16x!tt.ptr<bf16>>
    tt.store %8, %11 : tensor<16x16x!tt.ptr<bf16>>
    tt.return
  }
}
