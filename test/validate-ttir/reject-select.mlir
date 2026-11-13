// RUN: triton-atlas-opt %s --split-input-file --ttatlas-validate-ttir --verify-diagnostics

// Float compares, selects, and boolean or and xor: no single Atlas op does them.

tt.func @cmpf(%x: tensor<32x32xf32>) {
  // expected-error @below {{atlas: unsupported arith.cmpf: no direct Atlas equivalent: float compares, selects, and boolean or and xor}}
  // expected-error @below {{failed to legalize operation 'arith.cmpf' that was explicitly marked illegal}}
  %r = arith.cmpf olt, %x, %x : tensor<32x32xf32>
  tt.return
}

// -----

tt.func @select(%c: tensor<32x32xi1>, %x: tensor<32x32xf32>) {
  // expected-error @below {{atlas: unsupported arith.select: no direct Atlas equivalent: float compares, selects, and boolean or and xor}}
  // expected-error @below {{failed to legalize operation 'arith.select' that was explicitly marked illegal}}
  %r = arith.select %c, %x, %x : tensor<32x32xi1>, tensor<32x32xf32>
  tt.return
}

// -----

tt.func @ori(%a: tensor<32x32xi1>) {
  // expected-error @below {{atlas: unsupported arith.ori: no direct Atlas equivalent: float compares, selects, and boolean or and xor}}
  // expected-error @below {{failed to legalize operation 'arith.ori' that was explicitly marked illegal}}
  %r = arith.ori %a, %a : tensor<32x32xi1>
  tt.return
}

// -----

tt.func @xori(%a: tensor<32x32xi1>) {
  // expected-error @below {{atlas: unsupported arith.xori: no direct Atlas equivalent: float compares, selects, and boolean or and xor}}
  // expected-error @below {{failed to legalize operation 'arith.xori' that was explicitly marked illegal}}
  %r = arith.xori %a, %a : tensor<32x32xi1>
  tt.return
}
