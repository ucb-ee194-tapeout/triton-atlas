// RUN: triton-atlas-opt %s --canonicalize --cse | FileCheck %s

// The pipeline runs canonicalize and CSE after passes 2, 3 and 4, and later passes read the tags earlier passes
// set. This checks the tags on loads, stores and dots survive both, and that CSE keeps two loads with different
// slots apart while merging two identical ones.

#o = affine_map<()[s0, s1, s2, s3, s4, s5] -> (0, 0)>
tt.func @k(%p: !tt.ptr<f8E4M3FN>, %q: !tt.ptr<bf16>) {
  %s = tt.splat %p : !tt.ptr<f8E4M3FN> -> tensor<32x32x!tt.ptr<f8E4M3FN>>
  // CHECK: tt.load {{.*}}ttatlas.host_dtype = f8E4M3FN, ttatlas.slot = #ttatlas.slot<index = 0
  %a = tt.load %s {ttatlas.host_dtype = f8E4M3FN, ttatlas.slot = #ttatlas.slot<index = 0, arg = 0, role = activation_fp8, tile_origin = #o, logical_shape = [32, 32], host_dtype = f8E4M3FN>} : tensor<32x32x!tt.ptr<f8E4M3FN>>
  // CHECK: tt.load {{.*}}ttatlas.slot = #ttatlas.slot<index = 1
  %b = tt.load %s {ttatlas.host_dtype = f8E4M3FN, ttatlas.slot = #ttatlas.slot<index = 1, arg = 0, role = weight_fp8_kn, tile_origin = #o, logical_shape = [32, 32], host_dtype = f8E4M3FN>} : tensor<32x32x!tt.ptr<f8E4M3FN>>
  // Identical to %b, tags included: CSE may merge it into %b, which gives one slot.
  %b2 = tt.load %s {ttatlas.host_dtype = f8E4M3FN, ttatlas.slot = #ttatlas.slot<index = 1, arg = 0, role = weight_fp8_kn, tile_origin = #o, logical_shape = [32, 32], host_dtype = f8E4M3FN>} : tensor<32x32x!tt.ptr<f8E4M3FN>>
  // CHECK-NOT: tt.load
  %zero = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
  // CHECK: tt.dot {{.*}}ttatlas.k_chain = #ttatlas.k_chain<id = 0, pos = 0, len = 2>, ttatlas.mxu = 0 : i32
  %d0 = tt.dot %a, %b, %zero {ttatlas.k_chain = #ttatlas.k_chain<id = 0, pos = 0, len = 2>, ttatlas.mxu = 0 : i32} : tensor<32x32xf8E4M3FN> * tensor<32x32xf8E4M3FN> -> tensor<32x32xf32>
  // CHECK: tt.dot {{.*}}ttatlas.k_chain = #ttatlas.k_chain<id = 0, pos = 1, len = 2>, ttatlas.mxu = 0 : i32
  %d1 = tt.dot %a, %b2, %d0 {ttatlas.k_chain = #ttatlas.k_chain<id = 0, pos = 1, len = 2>, ttatlas.mxu = 0 : i32} : tensor<32x32xf8E4M3FN> * tensor<32x32xf8E4M3FN> -> tensor<32x32xf32>
  %o16 = arith.truncf %d1 : tensor<32x32xf32> to tensor<32x32xbf16>
  %t = tt.splat %q : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
  // CHECK: tt.store {{.*}}ttatlas.host_dtype = f32, ttatlas.slot = #ttatlas.slot<index = 0, arg = 1, role = output_bf16
  tt.store %t, %o16 {ttatlas.host_dtype = f32, ttatlas.slot = #ttatlas.slot<index = 0, arg = 1, role = output_bf16, tile_origin = #o, logical_shape = [32, 32], host_dtype = f32>} : tensor<32x32x!tt.ptr<bf16>>
  tt.return
}
