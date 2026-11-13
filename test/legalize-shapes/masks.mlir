// RUN: triton-atlas-opt %s --split-input-file --verify-diagnostics --ttatlas-legalize-shapes | FileCheck %s

// Masks after pass 1, which lets every mask through, so pass 4 decides: a tail mask becomes padding, any other is
// rejected with a reason.

// A tail mask as Triton itself emits it: the bound folded into a dense constant (Triton's canonicalizer turns
// tl.arange(0, 32) < 16 into arith.constant dense<16> : tensor<32xi32>), not a tt.splat of a scalar. The loaded tile is
// padded like legalize-shapes.mlir's @masked_tile, which uses the splat form.
// CHECK-LABEL: tt.func @triton_bound
// CHECK: tt.load {{.*}}ttatlas.pad = #ttatlas.pad<neutral = zero, logical_shape = [16, 32]>
tt.func @triton_bound(%ptr: !tt.ptr<f32>) {
  %ptrs = tt.splat %ptr : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>>
  %range = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %limits = arith.constant dense<16> : tensor<32xi32>
  %cmp = arith.cmpi slt, %range, %limits : tensor<32xi32>
  %row_mask = tt.expand_dims %cmp {axis = 1 : i32} : tensor<32xi1> -> tensor<32x1xi1>
  %mask = tt.broadcast %row_mask : tensor<32x1xi1> -> tensor<32x32xi1>
  %zero = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
  %a = tt.load %ptrs, %mask, %zero : tensor<32x32x!tt.ptr<f32>>
  %b = tt.load %ptrs : tensor<32x32x!tt.ptr<f32>>
  %empty = tensor.empty() : tensor<32x32xf32>
  %scalar_zero = arith.constant 0.0 : f32
  %acc = linalg.fill ins(%scalar_zero : f32) outs(%empty : tensor<32x32xf32>) -> tensor<32x32xf32>
  %result = linalg.matmul ins(%a, %b : tensor<32x32xf32>, tensor<32x32xf32>)
                          outs(%acc : tensor<32x32xf32>) -> tensor<32x32xf32>
  tt.store %ptrs, %result, %mask : tensor<32x32x!tt.ptr<f32>>
  tt.return
}

// -----

// A compare that isn't make_range < N: offs > 20 masks the head, not the tail.
tt.func @head_mask(%p: !tt.ptr<bf16>) {
  %c20 = arith.constant 20 : i32
  %c20s = tt.splat %c20 : i32 -> tensor<32xi32>
  %zero = arith.constant dense<0.0> : tensor<32xbf16>
  %r = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %m = arith.cmpi sgt, %r, %c20s : tensor<32xi32>
  %ps = tt.splat %p : !tt.ptr<bf16> -> tensor<32x!tt.ptr<bf16>>
  %a = tt.addptr %ps, %r : tensor<32x!tt.ptr<bf16>>, tensor<32xi32>
  // expected-error @below {{edge mask is not a static make_range comparison against a constant}}
  %x = tt.load %a, %m, %zero : tensor<32x!tt.ptr<bf16>>
  tt.return
}

// -----

// A masked load with no `other`: its masked-off lanes are undefined, so they hold no neutral element.
tt.func @no_other(%p: !tt.ptr<bf16>) {
  %c20 = arith.constant 20 : i32
  %c20s = tt.splat %c20 : i32 -> tensor<32xi32>
  %r = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %m = arith.cmpi slt, %r, %c20s : tensor<32xi32>
  %ps = tt.splat %p : !tt.ptr<bf16> -> tensor<32x!tt.ptr<bf16>>
  %a = tt.addptr %ps, %r : tensor<32x!tt.ptr<bf16>>, tensor<32xi32>
  // expected-error @below {{masked load has no explicit neutral value for padded lanes}}
  %x = tt.load %a, %m : tensor<32x!tt.ptr<bf16>>
  tt.return
}
