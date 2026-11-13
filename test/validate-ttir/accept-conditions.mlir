// RUN: triton-atlas-opt %s --split-input-file --ttatlas-validate-ttir | FileCheck %s

// The accepted side of each condition on a legal op (the reject-*.mlir files test the other side). Each kernel
// passes unchanged, apart from the host dtype tags on loads and stores.

// Unused scalar arguments (Inductor passes sizes it doesn't read), an integer one used only by tl.assume, and an
// f32 output (the driver converts Atlas's BF16).
// CHECK-LABEL: tt.func @unused_scalar
// CHECK: tt.store %{{.*}}, %{{.*}} {ttatlas.host_dtype = f32}
tt.func @unused_scalar(%eps: f32, %p: !tt.ptr<f32>, %n: i32, %stride: i32) {
  %c0 = arith.constant 0 : i32
  %pos = arith.cmpi sgt, %stride, %c0 : i32
  llvm.intr.assume %pos : i1
  %r = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %ps = tt.splat %p : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
  %a = tt.addptr %ps, %r : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
  %v = arith.constant dense<0.000000e+00> : tensor<32xf32>
  tt.store %a, %v : tensor<32x!tt.ptr<f32>>
  tt.return
}

// -----

// BF16 <-> f32 conversions, as arith.extf, arith.truncf and tt.fp_to_fp.
// CHECK-LABEL: tt.func @conversions
// CHECK: tt.load %{{.*}} {ttatlas.host_dtype = bf16}
// CHECK: arith.extf
// CHECK: arith.truncf
// CHECK: tt.fp_to_fp
tt.func @conversions(%p: !tt.ptr<bf16>) {
  %r = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %ps = tt.splat %p : !tt.ptr<bf16> -> tensor<32x!tt.ptr<bf16>>
  %a = tt.addptr %ps, %r : tensor<32x!tt.ptr<bf16>>, tensor<32xi32>
  %x = tt.load %a : tensor<32x!tt.ptr<bf16>>
  %f = arith.extf %x : tensor<32xbf16> to tensor<32xf32>
  %b = arith.truncf %f : tensor<32xf32> to tensor<32xbf16>
  %g = tt.fp_to_fp %b : tensor<32xbf16> -> tensor<32xf32>
  tt.return
}

// -----

// 1 / x (Atlas's reciprocal table).
// CHECK-LABEL: tt.func @reciprocal
// CHECK: arith.divf
tt.func @reciprocal(%x: tensor<32x32xf32>) {
  %one = arith.constant dense<1.000000e+00> : tensor<32x32xf32>
  %r = arith.divf %one, %x : tensor<32x32xf32>
  tt.return
}

// -----

// A row reduction (axis 1) with a max combine.
// CHECK-LABEL: tt.func @row_max
// CHECK: tt.reduce
tt.func @row_max(%x: tensor<32x32xf32>) {
  %r = "tt.reduce"(%x) <{axis = 1 : i32}> ({
  ^bb0(%a: f32, %b: f32):
    %m = arith.maxnumf %a, %b : f32
    tt.reduce.return %m : f32
  }) : (tensor<32x32xf32>) -> tensor<32xf32>
  tt.return
}

// -----

// An FP8 transpose feeding a dot's second operand (the weight), FP8 x FP8 into f32.
// CHECK-LABEL: tt.func @trans_dot
// CHECK: tt.trans
// CHECK: tt.dot
tt.func @trans_dot(%x: tensor<32x32xf8E4M3FN>, %w: tensor<32x32xf8E4M3FN>) {
  %zero = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
  %wt = tt.trans %w {order = array<i32: 1, 0>} : tensor<32x32xf8E4M3FN> -> tensor<32x32xf8E4M3FN>
  %d = tt.dot %x, %wt, %zero : tensor<32x32xf8E4M3FN> * tensor<32x32xf8E4M3FN> -> tensor<32x32xf32>
  tt.return
}

// -----

// An unused ub.poison, and a loop with a constant trip count (pass 2 unrolls it).
// CHECK-LABEL: tt.func @poison_loop
// CHECK: ub.poison
// CHECK: scf.for
tt.func @poison_loop(%x: tensor<32x32xf32>) {
  %p = ub.poison : tensor<32x32xf32>
  %c0 = arith.constant 0 : i32
  %c4 = arith.constant 4 : i32
  %c1 = arith.constant 1 : i32
  %s = scf.for %k = %c0 to %c4 step %c1 iter_args(%acc = %x) -> (tensor<32x32xf32>) : i32 {
    %n = arith.addf %acc, %x : tensor<32x32xf32>
    scf.yield %n : tensor<32x32xf32>
  }
  tt.return
}
