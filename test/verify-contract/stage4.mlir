// RUN: triton-atlas-opt %s --split-input-file --verify-diagnostics --ttatlas-verify-contract=stage=4

// Pass 4's contract: compute is linalg on single tiles (32 or 32x32); the memory boundary is one linalg.pack of each
// tt.load (through float casts) and one linalg.unpack read only by a tt.store (through float casts); no tile loop is left. A kernel with no compute keeps its tt.load feeding its
// tt.store.

tt.func @ok(%in: tensor<64x64x!tt.ptr<bf16>>, %out: tensor<64x64x!tt.ptr<bf16>>) {
  %x = tt.load %in : tensor<64x64x!tt.ptr<bf16>>
  %e = tensor.empty() : tensor<2x2x32x32xbf16>
  %p = linalg.pack %x inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %e : tensor<64x64xbf16> -> tensor<2x2x32x32xbf16>
  %t = tensor.extract_slice %p[0, 0, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<2x2x32x32xbf16> to tensor<32x32xbf16>
  %c = tensor.empty() : tensor<32x32xbf16>
  %s = linalg.add ins(%t, %t : tensor<32x32xbf16>, tensor<32x32xbf16>) outs(%c : tensor<32x32xbf16>) -> tensor<32x32xbf16>
  %big = tensor.insert_slice %s into %p[0, 0, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<32x32xbf16> into tensor<2x2x32x32xbf16>
  %o = tensor.empty() : tensor<64x64xbf16>
  %u = linalg.unpack %big inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %o : tensor<2x2x32x32xbf16> -> tensor<64x64xbf16>
  tt.store %out, %u : tensor<64x64x!tt.ptr<bf16>>
  tt.return
}

// -----

// A plain copy has no compute, so it keeps its load and store.
tt.func @copy(%in: tensor<64x64x!tt.ptr<bf16>>, %out: tensor<64x64x!tt.ptr<bf16>>) {
  %x = tt.load %in : tensor<64x64x!tt.ptr<bf16>>
  tt.store %out, %x : tensor<64x64x!tt.ptr<bf16>>
  tt.return
}

// -----

tt.func @big_linalg(%x: tensor<64x64xbf16>) {
  %e = tensor.empty() : tensor<64x64xbf16>
  // expected-error @below {{contract of pass 4 broken: linalg value is not one tile (32 or 32x32)}}
  %s = linalg.add ins(%x, %x : tensor<64x64xbf16>, tensor<64x64xbf16>) outs(%e : tensor<64x64xbf16>) -> tensor<64x64xbf16>
  tt.return
}

// -----

// An unpack read by compute: the pair was not folded.
tt.func @unfolded_pair(%in: tensor<32x32x!tt.ptr<bf16>>) {
  %x = tt.load %in : tensor<32x32x!tt.ptr<bf16>>
  %e = tensor.empty() : tensor<1x1x32x32xbf16>
  %p = linalg.pack %x inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %e : tensor<32x32xbf16> -> tensor<1x1x32x32xbf16>
  %o = tensor.empty() : tensor<32x32xbf16>
  // expected-error @below {{contract of pass 4 broken: linalg.unpack read by something other than a tt.store}}
  %u = linalg.unpack %p inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %o : tensor<1x1x32x32xbf16> -> tensor<32x32xbf16>
  %e2 = tensor.empty() : tensor<1x1x32x32xbf16>
  // expected-error @below {{contract of pass 4 broken: linalg.pack of something other than a tt.load}}
  %p2 = linalg.pack %u inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %e2 : tensor<32x32xbf16> -> tensor<1x1x32x32xbf16>
  tt.return
}

// -----

// A pack of a constant or an empty tensor was not folded.
tt.func @pack_of_constant() {
  %c = arith.constant dense<0.0> : tensor<32x32xbf16>
  %e = tensor.empty() : tensor<1x1x32x32xbf16>
  // expected-error @below {{contract of pass 4 broken: linalg.pack of something other than a tt.load}}
  %p = linalg.pack %c inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %e : tensor<32x32xbf16> -> tensor<1x1x32x32xbf16>
  tt.return
}

// -----

// A load read by compute directly, with no pack.
tt.func @load_unpacked(%in: tensor<32x32x!tt.ptr<bf16>>) {
  // expected-error @below {{contract of pass 4 broken: tt.load read by something other than a linalg.pack}}
  %x = tt.load %in : tensor<32x32x!tt.ptr<bf16>>
  %e = tensor.empty() : tensor<32x32xbf16>
  %s = linalg.add ins(%x, %x : tensor<32x32xbf16>, tensor<32x32xbf16>) outs(%e : tensor<32x32xbf16>) -> tensor<32x32xbf16>
  tt.return
}

// -----

// A tile loop left by tiling (no unroll).
tt.func @loop_left(%x: tensor<32x32xbf16>) {
  %lo = arith.constant 0 : index
  %hi = arith.constant 2 : index
  %one = arith.constant 1 : index
  // expected-error @below {{contract of pass 4 broken: tile loop left}}
  %r = scf.for %i = %lo to %hi step %one iter_args(%a = %x) -> tensor<32x32xbf16> {
    scf.yield %a : tensor<32x32xbf16>
  }
  tt.return
}
