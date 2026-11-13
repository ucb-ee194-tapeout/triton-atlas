// RUN: triton-atlas-opt %s --split-input-file --ttatlas-validate-ttir --verify-diagnostics

// tt.extern_elementwise: external functions have no Atlas meaning yet.

tt.func @extern(%x: tensor<32x32xf32>, %i: tensor<32x32xi32>, %c: i1, %p: !tt.ptr<f32>) {
  // expected-error @below {{atlas: unsupported tt.extern_elementwise: no lowering yet: external library functions have no Atlas pattern}}
  // expected-error @below {{failed to legalize operation 'tt.extern_elementwise' that was explicitly marked illegal}}
  %r = tt.extern_elementwise %x {libname = "", libpath = "", pure = true, symbol = "__nv_erff"} : (tensor<32x32xf32>) -> tensor<32x32xf32>
  tt.return
}
