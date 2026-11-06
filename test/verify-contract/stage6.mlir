// RUN: triton-atlas-opt %s --split-input-file --verify-diagnostics --ttatlas-verify-contract=stage=6

// Pass 6's contract: every tt.load / tt.store carries #ttatlas.slot.

tt.func @ok(%p: !tt.ptr<bf16>) {
  %s = tt.splat %p : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
  %v = tt.load %s {ttatlas.slot = #ttatlas.slot<index = 0, arg = 0, role = bias_bf16, tile_origin = affine_map<()[s0, s1, s2, s3, s4, s5] -> (0, 0)>, logical_shape = [32, 32], host_dtype = bf16>} : tensor<32x32x!tt.ptr<bf16>>
  tt.store %s, %v {ttatlas.slot = #ttatlas.slot<index = 0, arg = 0, role = output_bf16, tile_origin = affine_map<()[s0, s1, s2, s3, s4, s5] -> (0, 0)>, logical_shape = [32, 32], host_dtype = bf16>} : tensor<32x32x!tt.ptr<bf16>>
  tt.return
}

// -----

tt.func @no_slot(%p: !tt.ptr<bf16>) {
  %s = tt.splat %p : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
  // expected-error @below {{contract of pass 6 broken: missing #ttatlas.slot}}
  %v = tt.load %s : tensor<32x32x!tt.ptr<bf16>>
  tt.return
}
