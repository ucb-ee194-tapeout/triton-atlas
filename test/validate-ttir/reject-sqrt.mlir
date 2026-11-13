// RUN: triton-atlas-opt %s --split-input-file --ttatlas-validate-ttir --verify-diagnostics

// math.sqrt and math.rsqrt: no pattern yet for Atlas's square root table.

tt.func @sqrt(%x: tensor<32x32xf32>, %i: tensor<32x32xi32>, %c: i1, %p: !tt.ptr<f32>) {
  // expected-error @below {{atlas: unsupported math.sqrt: no lowering yet: Atlas's square root table has no pattern (rsqrt later as 1 / sqrt)}}
  // expected-error @below {{failed to legalize operation 'math.sqrt' that was explicitly marked illegal}}
  %r = math.sqrt %x : tensor<32x32xf32>
  tt.return
}
