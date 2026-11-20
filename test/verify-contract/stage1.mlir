// RUN: triton-atlas-opt %s --split-input-file --verify-diagnostics --ttatlas-verify-contract=stage=1

// ttatlas-verify-contract checks the output contract of one pass (here pass 1: one tt.func). Each case keeps or
// breaks that promise.

tt.func @ok(%p: !tt.ptr<bf16>) {
  %s = tt.splat %p : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
  %v = tt.load %s : tensor<32x32x!tt.ptr<bf16>>
  tt.store %s, %v : tensor<32x32x!tt.ptr<bf16>>
  tt.return
}

// -----

// expected-error @below {{contract of pass 1 broken: expected one tt.func, found 2}}
module {
  tt.func @a() { tt.return }
  tt.func @b() { tt.return }
}
