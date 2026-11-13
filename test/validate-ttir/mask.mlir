// RUN: triton-atlas-opt %s --split-input-file --ttatlas-validate-ttir | FileCheck %s

// Masks (arith.cmpi, arith.andi, a mask operand on a load or store) pass pass 1: pass 4 turns a mask with compile-time
// bounds into padding and rejects any other (a runtime bound, a compare that isn't offs < N, a load without `other`),
// so the rules on masks live there (test/legalize-shapes).

// A tail mask, offs < 20, as tl.load(..., mask=offs < 20) builds it.
// CHECK-LABEL: tt.func @tail
// CHECK: arith.cmpi slt
// CHECK: tt.load %{{.*}}, %{{.*}}
tt.func @tail(%p: !tt.ptr<bf16>) {
  %c20 = arith.constant dense<20> : tensor<32xi32>
  %r = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %m = arith.cmpi slt, %r, %c20 : tensor<32xi32>
  %ps = tt.splat %p : !tt.ptr<bf16> -> tensor<32x!tt.ptr<bf16>>
  %a = tt.addptr %ps, %r : tensor<32x!tt.ptr<bf16>>, tensor<32xi32>
  %x = tt.load %a, %m : tensor<32x!tt.ptr<bf16>>
  tt.return
}

// -----

// Two masks combined, as a 2-D tile's row and column masks are.
// CHECK-LABEL: tt.func @both
// CHECK: arith.andi
tt.func @both(%rows: tensor<32x32xi1>, %cols: tensor<32x32xi1>) {
  %m = arith.andi %rows, %cols : tensor<32x32xi1>
  tt.return
}

// -----

// A mask that isn't built from cmpi (here a constant) passes too; pass 4 decides.
// CHECK-LABEL: tt.func @constant_mask
// CHECK: tt.load %{{.*}}, %{{.*}}
tt.func @constant_mask(%p: !tt.ptr<bf16>) {
  %m = arith.constant dense<[true, false, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true]> : tensor<32xi1>
  %r = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %ps = tt.splat %p : !tt.ptr<bf16> -> tensor<32x!tt.ptr<bf16>>
  %a = tt.addptr %ps, %r : tensor<32x!tt.ptr<bf16>>, tensor<32xi32>
  %x = tt.load %a, %m : tensor<32x!tt.ptr<bf16>>
  tt.return
}

// -----

// CHECK-LABEL: tt.func @masked_store
// CHECK: tt.store %{{.*}}, %{{.*}}, %{{.*}}
tt.func @masked_store(%p: !tt.ptr<bf16>, %x: tensor<32xbf16>) {
  %m = arith.constant dense<true> : tensor<32xi1>
  %r = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %ps = tt.splat %p : !tt.ptr<bf16> -> tensor<32x!tt.ptr<bf16>>
  %a = tt.addptr %ps, %r : tensor<32x!tt.ptr<bf16>>, tensor<32xi32>
  tt.store %a, %x, %m : tensor<32x!tt.ptr<bf16>>
  tt.return
}

// -----

// A compare on a constant-trip loop's counter (an Inductor-style tail test, k < 100) is a mask too.
// CHECK-LABEL: tt.func @loop_counter
// CHECK: arith.cmpi slt
tt.func @loop_counter(%p: !tt.ptr<bf16>) {
  %r = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %ps = tt.splat %p : !tt.ptr<bf16> -> tensor<32x!tt.ptr<bf16>>
  %a = tt.addptr %ps, %r : tensor<32x!tt.ptr<bf16>>, tensor<32xi32>
  %c0 = arith.constant 0 : i32
  %c4 = arith.constant 4 : i32
  %c1 = arith.constant 1 : i32
  %c100 = arith.constant dense<100> : tensor<32xi32>
  scf.for %k = %c0 to %c4 step %c1 : i32 {
    %ks = tt.splat %k : i32 -> tensor<32xi32>
        %m = arith.cmpi slt, %ks, %c100 : tensor<32xi32>
    %v = tt.load %a, %m : tensor<32x!tt.ptr<bf16>>
  }
  tt.return
}
