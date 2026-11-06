// RUN: triton-atlas-opt %s --split-input-file --verify-diagnostics --ttatlas-verify-contract=stage=3

// Pass 3's contract: compute on float tensors is linalg; constants, loads, stores and integer or pointer address math
// stay TTIR.

tt.func @ok(%p: !tt.ptr<f8E4M3FN>) {
  %r = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %e = tt.expand_dims %r {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
  %b = tt.broadcast %e : tensor<1x32xi32> -> tensor<32x32xi32>
  %s = tt.splat %p : !tt.ptr<f8E4M3FN> -> tensor<32x32x!tt.ptr<f8E4M3FN>>
  %a = tt.addptr %s, %b : tensor<32x32x!tt.ptr<f8E4M3FN>>, tensor<32x32xi32>
  %x = tt.load %a : tensor<32x32x!tt.ptr<f8E4M3FN>>
  %zero = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<32x32xf32>
  %acc = linalg.fill ins(%zero : f32) outs(%empty : tensor<32x32xf32>) -> tensor<32x32xf32>
  %m = linalg.matmul ins(%x, %x : tensor<32x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>) outs(%acc : tensor<32x32xf32>) -> tensor<32x32xf32>
  tt.return
}

// -----

tt.func @dot_left(%a: tensor<32x32xf8E4M3FN>, %b: tensor<32x32xf8E4M3FN>) {
  %zero = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
  // expected-error @below {{contract of pass 3 broken: TTIR compute is left: tt.dot}}
  %d = tt.dot %a, %b, %zero : tensor<32x32xf8E4M3FN> * tensor<32x32xf8E4M3FN> -> tensor<32x32xf32>
  tt.return
}

// -----

tt.func @elementwise_left(%x: tensor<32x32xbf16>) {
  // expected-error @below {{contract of pass 3 broken: TTIR compute is left: arith.addf}}
  %y = arith.addf %x, %x : tensor<32x32xbf16>
  tt.return
}

// -----

// A broadcast of float data is compute (linalg.broadcast), unlike a broadcast of offsets.
tt.func @float_broadcast_left(%x: tensor<1x32xbf16>) {
  // expected-error @below {{contract of pass 3 broken: TTIR compute is left: tt.broadcast}}
  %y = tt.broadcast %x : tensor<1x32xbf16> -> tensor<32x32xbf16>
  tt.return
}

// -----

tt.func @math_left(%x: tensor<32x32xbf16>) {
  // expected-error @below {{contract of pass 3 broken: TTIR compute is left: math.exp}}
  %y = math.exp %x : tensor<32x32xbf16>
  tt.return
}

// -----

// A comparison of float tensors is compute too, though its result is i1.
tt.func @cmpf_left(%x: tensor<32x32xbf16>) {
  // expected-error @below {{contract of pass 3 broken: TTIR compute is left: arith.cmpf}}
  %y = arith.cmpf ogt, %x, %x : tensor<32x32xbf16>
  tt.return
}

// -----

tt.func @trans_left(%x: tensor<32x32xf8E4M3FN>) {
  // expected-error @below {{contract of pass 3 broken: TTIR compute is left: tt.trans}}
  %y = tt.trans %x {order = array<i32: 1, 0>} : tensor<32x32xf8E4M3FN> -> tensor<32x32xf8E4M3FN>
  tt.return
}

// -----

tt.func @reduce_left(%x: tensor<32x32xbf16>) {
  // expected-error @below {{contract of pass 3 broken: TTIR compute is left: tt.reduce}}
  %y = "tt.reduce"(%x) <{axis = 1 : i32}> ({
  ^bb0(%a: bf16, %b: bf16):
    %m = arith.maxnumf %a, %b : bf16
    tt.reduce.return %m : bf16
  }) : (tensor<32x32xbf16>) -> tensor<32xbf16>
  tt.return
}

// -----

tt.func @float_expand_dims_left(%x: tensor<32xbf16>) {
  // expected-error @below {{contract of pass 3 broken: TTIR compute is left: tt.expand_dims}}
  %y = tt.expand_dims %x {axis = 0 : i32} : tensor<32xbf16> -> tensor<1x32xbf16>
  tt.return
}
