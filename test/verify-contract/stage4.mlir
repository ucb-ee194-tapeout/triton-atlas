// RUN: triton-atlas-opt %s --split-input-file --verify-diagnostics --ttatlas-verify-contract=stage=4

// Pass 4's contract: every linalg op works on 32x32 tensors; every linalg.matmul carries ttatlas.mxu; a larger tt.load
// is read only through constant 32x32 slices, and a larger stored tensor is built only from constant 32x32
// insert_slices over a tensor.empty.

tt.func @ok(%in: tensor<64x64x!tt.ptr<bf16>>, %out: tensor<64x64x!tt.ptr<bf16>>) {
  %x = tt.load %in : tensor<64x64x!tt.ptr<bf16>>
  %t00 = tensor.extract_slice %x[0, 0] [32, 32] [1, 1] : tensor<64x64xbf16> to tensor<32x32xbf16>
  %t01 = tensor.extract_slice %x[0, 32] [32, 32] [1, 1] : tensor<64x64xbf16> to tensor<32x32xbf16>
  %e = tensor.empty() : tensor<32x32xbf16>
  %s = linalg.add ins(%t00, %t01 : tensor<32x32xbf16>, tensor<32x32xbf16>) outs(%e : tensor<32x32xbf16>) -> tensor<32x32xbf16>
  %big = tensor.empty() : tensor<64x64xbf16>
  %y0 = tensor.insert_slice %s into %big[0, 0] [32, 32] [1, 1] : tensor<32x32xbf16> into tensor<64x64xbf16>
  %y1 = tensor.insert_slice %s into %y0[0, 32] [32, 32] [1, 1] : tensor<32x32xbf16> into tensor<64x64xbf16>
  %y2 = tensor.insert_slice %s into %y1[32, 0] [32, 32] [1, 1] : tensor<32x32xbf16> into tensor<64x64xbf16>
  %y3 = tensor.insert_slice %s into %y2[32, 32] [32, 32] [1, 1] : tensor<32x32xbf16> into tensor<64x64xbf16>
  tt.store %out, %y3 : tensor<64x64x!tt.ptr<bf16>>
  tt.return
}

// -----

tt.func @big_linalg(%x: tensor<64x64xbf16>) {
  %e = tensor.empty() : tensor<64x64xbf16>
  // expected-error @below {{contract of pass 4 broken: 2-D linalg value is not 32x32}}
  %s = linalg.add ins(%x, %x : tensor<64x64xbf16>, tensor<64x64xbf16>) outs(%e : tensor<64x64xbf16>) -> tensor<64x64xbf16>
  tt.return
}

// -----

tt.func @no_unit(%a: tensor<32x32xf8E4M3FN>, %acc: tensor<32x32xf32>) {
  // expected-error @below {{contract of pass 4 broken: linalg.matmul without ttatlas.mxu}}
  %m = linalg.matmul ins(%a, %a : tensor<32x32xf8E4M3FN>, tensor<32x32xf8E4M3FN>) outs(%acc : tensor<32x32xf32>) -> tensor<32x32xf32>
  tt.return
}

// -----

// A 64x64 load used whole, not through 32x32 slices.
tt.func @load_used_whole(%in: tensor<64x64x!tt.ptr<bf16>>, %out: tensor<64x64x!tt.ptr<bf16>>) {
  // expected-error @below {{contract of pass 4 broken: tt.load is read other than through constant 32x32 slices}}
  %x = tt.load %in : tensor<64x64x!tt.ptr<bf16>>
  // expected-error @below {{contract of pass 4 broken: stored tensor is not built from constant 32x32 insert_slices}}
  tt.store %out, %x : tensor<64x64x!tt.ptr<bf16>>
  tt.return
}

// -----

// A 64x64 store of a value that is not assembled from 32x32 tiles.
tt.func @store_not_tiled(%out: tensor<64x64x!tt.ptr<bf16>>, %y: tensor<64x64xbf16>) {
  // expected-error @below {{contract of pass 4 broken: stored tensor is not built from constant 32x32 insert_slices}}
  tt.store %out, %y : tensor<64x64x!tt.ptr<bf16>>
  tt.return
}

// -----

// Three tiles of four: the stored tensor is not covered.
tt.func @store_missing_tile(%out: tensor<64x64x!tt.ptr<bf16>>, %t: tensor<32x32xbf16>) {
  %big = tensor.empty() : tensor<64x64xbf16>
  %y0 = tensor.insert_slice %t into %big[0, 0] [32, 32] [1, 1] : tensor<32x32xbf16> into tensor<64x64xbf16>
  %y1 = tensor.insert_slice %t into %y0[0, 32] [32, 32] [1, 1] : tensor<32x32xbf16> into tensor<64x64xbf16>
  %y2 = tensor.insert_slice %t into %y1[32, 0] [32, 32] [1, 1] : tensor<32x32xbf16> into tensor<64x64xbf16>
  // expected-error @below {{contract of pass 4 broken: stored tensor is not built from constant 32x32 insert_slices}}
  tt.store %out, %y2 : tensor<64x64x!tt.ptr<bf16>>
  tt.return
}

// -----

// A load read through a slice that is not one 32x32 tile.
tt.func @load_bad_slice(%in: tensor<64x64x!tt.ptr<bf16>>) {
  // expected-error @below {{contract of pass 4 broken: tt.load is read other than through constant 32x32 slices}}
  %x = tt.load %in : tensor<64x64x!tt.ptr<bf16>>
  %t = tensor.extract_slice %x[0, 0] [32, 64] [1, 1] : tensor<64x64xbf16> to tensor<32x64xbf16>
  tt.return
}
