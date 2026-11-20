// RUN: triton-atlas-opt %s --split-input-file --verify-diagnostics --ttatlas-verify-contract=stage=6

// Pass 6's contract: every linalg.pack and linalg.unpack carries its slots (one per tile, or the bare slot of a single
// tile), a tt.load / tt.store with no pack or unpack behind it (a copy) carries one, and a non-splat constant tile read
// by a generic carries constant slots.

tt.func @copy(%p: !tt.ptr<bf16>) {
  %s = tt.splat %p : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
  %v = tt.load %s {ttatlas.slot = #ttatlas.slot<index = 0, kind = memory, arg = 0, role = bias_bf16, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>} : tensor<32x32x!tt.ptr<bf16>>
  tt.store %s, %v {ttatlas.slot = #ttatlas.slot<index = 0, kind = memory, arg = 0, role = output_bf16, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>} : tensor<32x32x!tt.ptr<bf16>>
  tt.return
}

// -----

tt.func @copy_without_slot(%p: !tt.ptr<bf16>) {
  %s = tt.splat %p : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
  // expected-error @below {{contract of pass 6 broken: missing #ttatlas.slot}}
  %v = tt.load %s : tensor<32x32x!tt.ptr<bf16>>
  // expected-error @below {{contract of pass 6 broken: missing #ttatlas.slot}}
  tt.store %s, %v : tensor<32x32x!tt.ptr<bf16>>
  tt.return
}

// -----

tt.func @pack_without_slot(%p: !tt.ptr<bf16>, %q: !tt.ptr<bf16>) {
  %s = tt.splat %p : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
  %v = tt.load %s : tensor<32x32x!tt.ptr<bf16>>
  %e = tensor.empty() : tensor<1x1x32x32xbf16>
  // expected-error @below {{contract of pass 6 broken: missing #ttatlas.slot}}
  %pk = linalg.pack %v inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %e : tensor<32x32xbf16> -> tensor<1x1x32x32xbf16>
  %o = tensor.empty() : tensor<32x32xbf16>
  // expected-error @below {{contract of pass 6 broken: missing #ttatlas.slot}}
  %u = linalg.unpack %pk inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %o : tensor<1x1x32x32xbf16> -> tensor<32x32xbf16>
  %sq = tt.splat %q : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
  tt.store %sq, %u : tensor<32x32x!tt.ptr<bf16>>
  tt.return
}
