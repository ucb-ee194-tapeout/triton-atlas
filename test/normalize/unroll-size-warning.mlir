// RUN: triton-atlas-opt %s --ttatlas-normalize --verify-diagnostics

// A K loop of 128 steps unrolls to 128 dots, about 128 x 23 = 2944 program words, more than the 2048 one Atlas
// program may hold at our pin. Pass 2 still unrolls it and only warns: the real size is checked at the end.

// expected-warning @below {{ttatlas-normalize: unrolled to 128 dots, at least 2944 program words; Atlas runs at most 2048 words per program, so this kernel will likely fail the program-size check}}
tt.func @long_k(%a: tensor<32x32xf8E4M3FN>, %b: tensor<32x32xf8E4M3FN>) -> tensor<32x32xf32> {
  %zero = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
  %c0 = arith.constant 0 : i32
  %c128 = arith.constant 128 : i32
  %c1 = arith.constant 1 : i32
  %acc = scf.for %k = %c0 to %c128 step %c1 iter_args(%s = %zero) -> (tensor<32x32xf32>) : i32 {
    %d = tt.dot %a, %b, %s : tensor<32x32xf8E4M3FN> * tensor<32x32xf8E4M3FN> -> tensor<32x32xf32>
    scf.yield %d : tensor<32x32xf32>
  }
  tt.return %acc : tensor<32x32xf32>
}
