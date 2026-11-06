// RUN: triton-atlas-opt %s --split-input-file --verify-diagnostics --ttatlas-verify-contract=stage=2

// Pass 2's contract: every tt.dot accumulator is zero, except a later step of a ttatlas.k_chain, which takes the
// previous step's result; no scf.for with a constant trip count.

tt.func @ok(%a: tensor<32x32xf8E4M3FN>, %b: tensor<32x32xf8E4M3FN>) {
  %zero = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
  %d0 = tt.dot %a, %b, %zero {ttatlas.k_chain = #ttatlas.k_chain<id = 0, pos = 0, len = 2>} : tensor<32x32xf8E4M3FN> * tensor<32x32xf8E4M3FN> -> tensor<32x32xf32>
  %d1 = tt.dot %a, %b, %d0 {ttatlas.k_chain = #ttatlas.k_chain<id = 0, pos = 1, len = 2>} : tensor<32x32xf8E4M3FN> * tensor<32x32xf8E4M3FN> -> tensor<32x32xf32>
  tt.return
}

// -----

tt.func @bias_in_dot(%a: tensor<32x32xf8E4M3FN>, %b: tensor<32x32xf8E4M3FN>, %c: tensor<32x32xf32>) {
  // expected-error @below {{contract of pass 2 broken: tt.dot accumulator is neither zero nor the previous step of its ttatlas.k_chain}}
  %d = tt.dot %a, %b, %c : tensor<32x32xf8E4M3FN> * tensor<32x32xf8E4M3FN> -> tensor<32x32xf32>
  tt.return
}

// -----

tt.func @constant_loop() {
  %c0 = arith.constant 0 : i32
  %c4 = arith.constant 4 : i32
  %c1 = arith.constant 1 : i32
  // expected-error @below {{contract of pass 2 broken: scf.for with a constant trip count is left}}
  scf.for %i = %c0 to %c4 step %c1 : i32 {
  }
  tt.return
}

// -----

tt.func @wrong_chain(%a: tensor<32x32xf8E4M3FN>, %b: tensor<32x32xf8E4M3FN>) {
  %zero = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
  %d0 = tt.dot %a, %b, %zero {ttatlas.k_chain = #ttatlas.k_chain<id = 0, pos = 0, len = 2>} : tensor<32x32xf8E4M3FN> * tensor<32x32xf8E4M3FN> -> tensor<32x32xf32>
  // expected-error @below {{contract of pass 2 broken: ttatlas.k_chain step does not take the previous step's result}}
  %d1 = tt.dot %a, %b, %d0 {ttatlas.k_chain = #ttatlas.k_chain<id = 1, pos = 1, len = 2>} : tensor<32x32xf8E4M3FN> * tensor<32x32xf8E4M3FN> -> tensor<32x32xf32>
  tt.return
}

// -----

tt.func @skipped_pos(%a: tensor<32x32xf8E4M3FN>, %b: tensor<32x32xf8E4M3FN>) {
  %zero = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
  %d0 = tt.dot %a, %b, %zero {ttatlas.k_chain = #ttatlas.k_chain<id = 0, pos = 0, len = 3>} : tensor<32x32xf8E4M3FN> * tensor<32x32xf8E4M3FN> -> tensor<32x32xf32>
  // expected-error @below {{contract of pass 2 broken: ttatlas.k_chain step does not take the previous step's result}}
  %d2 = tt.dot %a, %b, %d0 {ttatlas.k_chain = #ttatlas.k_chain<id = 0, pos = 2, len = 3>} : tensor<32x32xf8E4M3FN> * tensor<32x32xf8E4M3FN> -> tensor<32x32xf32>
  tt.return
}

// -----

// A later step that starts again from zero breaks the chain.
tt.func @restart(%a: tensor<32x32xf8E4M3FN>, %b: tensor<32x32xf8E4M3FN>) {
  %zero = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
  %d0 = tt.dot %a, %b, %zero {ttatlas.k_chain = #ttatlas.k_chain<id = 0, pos = 0, len = 2>} : tensor<32x32xf8E4M3FN> * tensor<32x32xf8E4M3FN> -> tensor<32x32xf32>
  // expected-error @below {{contract of pass 2 broken: ttatlas.k_chain step does not take the previous step's result}}
  %d1 = tt.dot %a, %b, %zero {ttatlas.k_chain = #ttatlas.k_chain<id = 0, pos = 1, len = 2>} : tensor<32x32xf8E4M3FN> * tensor<32x32xf8E4M3FN> -> tensor<32x32xf32>
  tt.return
}

// -----

tt.func @wrong_len(%a: tensor<32x32xf8E4M3FN>, %b: tensor<32x32xf8E4M3FN>) {
  %zero = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
  %d0 = tt.dot %a, %b, %zero {ttatlas.k_chain = #ttatlas.k_chain<id = 0, pos = 0, len = 3>} : tensor<32x32xf8E4M3FN> * tensor<32x32xf8E4M3FN> -> tensor<32x32xf32>
  // expected-error @below {{contract of pass 2 broken: ttatlas.k_chain step does not take the previous step's result}}
  %d1 = tt.dot %a, %b, %d0 {ttatlas.k_chain = #ttatlas.k_chain<id = 0, pos = 1, len = 2>} : tensor<32x32xf8E4M3FN> * tensor<32x32xf8E4M3FN> -> tensor<32x32xf32>
  tt.return
}
