// RUN: triton-atlas-opt %s --split-input-file --ttatlas-validate-ttir --verify-diagnostics

// A runtime scalar argument that is used. A float one is data (a scale, an epsilon), and vpu_splat needs its value
// as constant bits. An integer one is a size, stride or bound, which needs specialization first: no pass evaluates
// a runtime offset yet. The condition is on the function, so the error points at it.

// expected-error @below {{atlas: unsupported tt.func: uses the runtime scalar argument #1 as data; vpu_splat needs constant bits}}
// expected-error @below {{failed to legalize operation 'tt.func' that was explicitly marked illegal}}
tt.func @scale(%x: tensor<32x32xf32>, %s: f32) {
  %b = tt.splat %s : f32 -> tensor<32x32xf32>
  %r = arith.mulf %x, %b : tensor<32x32xf32>
  tt.return
}

// -----

// A runtime stride in address math, as the tutorial matmul passes its strides.
// expected-error @below {{atlas: unsupported tt.func: uses the runtime scalar argument #1 (a size, stride or bound); only constant sizes and strides until specialization exists}}
// expected-error @below {{failed to legalize operation 'tt.func' that was explicitly marked illegal}}
tt.func @stride(%p: !tt.ptr<bf16>, %stride: i32) {
  %r = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %s = tt.splat %stride : i32 -> tensor<32xi32>
  %o = arith.muli %r, %s : tensor<32xi32>
  %ps = tt.splat %p : !tt.ptr<bf16> -> tensor<32x!tt.ptr<bf16>>
  %a = tt.addptr %ps, %o : tensor<32x!tt.ptr<bf16>>, tensor<32xi32>
  %x = tt.load %a : tensor<32x!tt.ptr<bf16>>
  tt.return
}
