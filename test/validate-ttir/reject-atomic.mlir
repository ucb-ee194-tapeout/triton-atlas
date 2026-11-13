// RUN: triton-atlas-opt %s --split-input-file --ttatlas-validate-ttir --verify-diagnostics

// Atomics: Atlas has no atomic or read-modify-write memory instruction.

tt.func @atomic(%x: tensor<32x32xf32>, %i: tensor<32x32xi32>, %c: i1, %p: !tt.ptr<f32>) {
  %ps = tt.splat %p : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>>
  // expected-error @below {{atlas: unsupported tt.atomic_rmw: no direct Atlas equivalent: no atomic or read-modify-write memory instruction (could be built from a load, the op and a store on one core)}}
  // expected-error @below {{failed to legalize operation 'tt.atomic_rmw' that was explicitly marked illegal}}
  %a = tt.atomic_rmw fadd, acq_rel, gpu, %ps, %x : (tensor<32x32x!tt.ptr<f32>>, tensor<32x32xf32>) -> tensor<32x32xf32>
  tt.return
}
