// RUN: triton-atlas-opt %s --split-input-file --ttatlas-validate-ttir --verify-diagnostics

// Descriptor kernels: Triton's descriptor rewrite leaves masks, i64 offsets and reshapes, not plain pointers.

tt.func @descriptor(%x: tensor<32x32xf32>, %i: tensor<32x32xi32>, %c: i1, %p: !tt.ptr<f32>) {
  %c32 = arith.constant 32 : i32
  %c1 = arith.constant 1 : i64
  %c32l = arith.constant 32 : i64
  // expected-error @below {{atlas: unsupported tt.make_tensor_descriptor: no lowering yet: descriptor kernels; Triton's descriptor rewrite leaves masks, i64 offsets and reshapes, not plain pointers}}
  // expected-error @below {{failed to legalize operation 'tt.make_tensor_descriptor' that was explicitly marked illegal}}
  %d = tt.make_tensor_descriptor %p, [%c32, %c32], [%c32l, %c1] : !tt.ptr<f32>, !tt.tensordesc<32x32xf32>
  tt.return
}
