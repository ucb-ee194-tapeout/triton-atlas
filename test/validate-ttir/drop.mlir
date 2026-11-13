// RUN: triton-atlas-opt %s --ttatlas-validate-ttir | FileCheck %s

// gpu.barrier and llvm.intr.assume are dropped, not rejected: one Atlas core has nothing to wait for, and an assume
// is only a hint. tl.assume(pid >= 0) builds its condition with arith.cmpi; that cmpi is kept (dead, for the cleanup
// after pass 2), not rejected as a mask. The rest of the kernel is kept.

// CHECK-LABEL: tt.func @dropped
// CHECK: arith.cmpi sge
// CHECK-NOT: gpu.barrier
// CHECK-NOT: llvm.intr.assume
// CHECK: arith.addf
// CHECK: tt.return
tt.func @dropped(%x: tensor<32x32xf32>) {
  %c0 = arith.constant 0 : i32
  %pid = tt.get_program_id x : i32
  %c = arith.cmpi sge, %pid, %c0 : i32
  llvm.intr.assume %c : i1
  gpu.barrier
  %r = arith.addf %x, %x : tensor<32x32xf32>
  tt.return
}
