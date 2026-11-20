// RUN: triton-atlas-opt %s --ttatlas-assign-slots | FileCheck %s

// A mask that keeps rows 4 and below (row >= 4): the valid box starts at row 4, inside the block, which is the whole
// tile; the host writes zeros above it. Load and store carry the same box.
// CHECK: tt.load {{.*}}ttatlas.slot = #ttatlas.slot<index = 0, kind = memory, arg = 0, role = constant_bf16, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [4, 0, 32, 32]
// CHECK: tt.store {{.*}}ttatlas.slot = #ttatlas.slot<index = 0, kind = memory, arg = 1, role = output_bf16, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [4, 0, 32, 32]
tt.func @lo(%p: !tt.ptr<bf16>, %q: !tt.ptr<bf16>) {
  %r = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
  %c32 = arith.constant dense<32> : tensor<32x1xi32>
  %rows = tt.expand_dims %r {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
  %rowoff = arith.muli %rows, %c32 : tensor<32x1xi32>
  %cols = tt.expand_dims %r {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
  %rowb = tt.broadcast %rowoff : tensor<32x1xi32> -> tensor<32x32xi32>
  %colb = tt.broadcast %cols : tensor<1x32xi32> -> tensor<32x32xi32>
  %off = arith.addi %rowb, %colb : tensor<32x32xi32>
  %lim = arith.constant dense<4> : tensor<32x1xi32>
  %m1 = arith.cmpi sge, %rows, %lim : tensor<32x1xi32>
  %mask = tt.broadcast %m1 : tensor<32x1xi1> -> tensor<32x32xi1>
  %s = tt.splat %p : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
  %a = tt.addptr %s, %off : tensor<32x32x!tt.ptr<bf16>>, tensor<32x32xi32>
  %v = tt.load %a, %mask : tensor<32x32x!tt.ptr<bf16>>
  %sq = tt.splat %q : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
  %b = tt.addptr %sq, %off : tensor<32x32x!tt.ptr<bf16>>, tensor<32x32xi32>
  tt.store %b, %v, %mask : tensor<32x32x!tt.ptr<bf16>>
  tt.return
}
