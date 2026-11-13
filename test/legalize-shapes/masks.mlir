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

// -----

// A masked load whose padded lanes change before they are reduced: other = 0 matches the add's neutral, but x + 1.0
// turns each padded lane into 1, and the row sum then counts them. Pass 4 must reject this (or re-mask before the sum),
// not tag the load as padded with zero.
#id = affine_map<(d0, d1) -> (d0, d1)>
tt.func @padded_lanes_change_before_sum(%ptr: !tt.ptr<f32>) -> tensor<32xf32> {
  %ptrs = tt.splat %ptr : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>>
  %range = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %limit = arith.constant 16 : i32
  %limits = tt.splat %limit : i32 -> tensor<32xi32>
  %cmp = arith.cmpi slt, %range, %limits : tensor<32xi32>
  %col_mask = tt.expand_dims %cmp {axis = 0 : i32} : tensor<32xi1> -> tensor<1x32xi1>
  %mask = tt.broadcast %col_mask : tensor<1x32xi1> -> tensor<32x32xi1>
  %zero = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
  // expected-error @below {{masked load}}
  %x = tt.load %ptrs, %mask, %zero : tensor<32x32x!tt.ptr<f32>>
  %one = arith.constant 1.0 : f32
  %e = tensor.empty() : tensor<32x32xf32>
  %plus = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]}
      ins(%x : tensor<32x32xf32>) outs(%e : tensor<32x32xf32>) {
  ^bb0(%in: f32, %out: f32):
    %s = arith.addf %in, %one : f32
    linalg.yield %s : f32
  } -> tensor<32x32xf32>
  %z = arith.constant 0.0 : f32
  %e1 = tensor.empty() : tensor<32xf32>
  %init = linalg.fill ins(%z : f32) outs(%e1 : tensor<32xf32>) -> tensor<32xf32>
  %sum = linalg.reduce ins(%plus : tensor<32x32xf32>) outs(%init : tensor<32xf32>) dimensions = [1]
  (%a: f32, %b: f32) {
    %t = arith.addf %a, %b : f32
    linalg.yield %t : f32
  }
  tt.return %sum : tensor<32xf32>
}

// -----

// A column tail mask, (c < 16)[None, :]: the padding is in dimension 1, so the logical shape is 32x16. (@triton_bound
// masks rows; the mask's dimension comes from its expand_dims, not from the make_range.)
// CHECK-LABEL: tt.func @column_mask
// CHECK: tt.load {{.*}}ttatlas.pad = #ttatlas.pad<neutral = zero, logical_shape = [32, 16]>
tt.func @column_mask(%ptr: !tt.ptr<f32>) {
  %ptrs = tt.splat %ptr : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>>
  %range = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %limit = arith.constant 16 : i32
  %limits = tt.splat %limit : i32 -> tensor<32xi32>
  %cmp = arith.cmpi slt, %range, %limits : tensor<32xi32>
  %col_mask = tt.expand_dims %cmp {axis = 0 : i32} : tensor<32xi1> -> tensor<1x32xi1>
  %mask = tt.broadcast %col_mask : tensor<1x32xi1> -> tensor<32x32xi1>
  %zero = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
  %a = tt.load %ptrs, %mask, %zero : tensor<32x32x!tt.ptr<f32>>
  %b = tt.load %ptrs : tensor<32x32x!tt.ptr<f32>>
  %empty = tensor.empty() : tensor<32x32xf32>
  %scalar_zero = arith.constant 0.0 : f32
  %acc = linalg.fill ins(%scalar_zero : f32) outs(%empty : tensor<32x32xf32>) -> tensor<32x32xf32>
  %result = linalg.matmul ins(%a, %b : tensor<32x32xf32>, tensor<32x32xf32>)
                          outs(%acc : tensor<32x32xf32>) -> tensor<32x32xf32>
  tt.return
}

// -----

// A row tail and a column tail together, (r < 16)[:, None] & (c < 20)[None, :]: logical shape 16x20.
// CHECK-LABEL: tt.func @row_and_column_mask
// CHECK: tt.load {{.*}}ttatlas.pad = #ttatlas.pad<neutral = zero, logical_shape = [16, 20]>
tt.func @row_and_column_mask(%ptr: !tt.ptr<f32>) {
  %ptrs = tt.splat %ptr : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>>
  %range = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %c16 = arith.constant 16 : i32
  %c20 = arith.constant 20 : i32
  %rows16 = tt.splat %c16 : i32 -> tensor<32xi32>
  %cols20 = tt.splat %c20 : i32 -> tensor<32xi32>
  %row_cmp = arith.cmpi slt, %range, %rows16 : tensor<32xi32>
  %col_cmp = arith.cmpi slt, %range, %cols20 : tensor<32xi32>
  %row_2d = tt.expand_dims %row_cmp {axis = 1 : i32} : tensor<32xi1> -> tensor<32x1xi1>
  %col_2d = tt.expand_dims %col_cmp {axis = 0 : i32} : tensor<32xi1> -> tensor<1x32xi1>
  %row_mask = tt.broadcast %row_2d : tensor<32x1xi1> -> tensor<32x32xi1>
  %col_mask = tt.broadcast %col_2d : tensor<1x32xi1> -> tensor<32x32xi1>
  %mask = arith.andi %row_mask, %col_mask : tensor<32x32xi1>
  %zero = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
  %a = tt.load %ptrs, %mask, %zero : tensor<32x32x!tt.ptr<f32>>
  %b = tt.load %ptrs : tensor<32x32x!tt.ptr<f32>>
  %empty = tensor.empty() : tensor<32x32xf32>
  %scalar_zero = arith.constant 0.0 : f32
  %acc = linalg.fill ins(%scalar_zero : f32) outs(%empty : tensor<32x32xf32>) -> tensor<32x32xf32>
  %result = linalg.matmul ins(%a, %b : tensor<32x32xf32>, tensor<32x32xf32>)
                          outs(%acc : tensor<32x32xf32>) -> tensor<32x32xf32>
  tt.return
}

// -----

// The other spelling of a row tail: the range expanded first, then compared, r[:, None] < 16. Logical shape 16x32.
// CHECK-LABEL: tt.func @expanded_range_mask
// CHECK: tt.load {{.*}}ttatlas.pad = #ttatlas.pad<neutral = zero, logical_shape = [16, 32]>
tt.func @expanded_range_mask(%ptr: !tt.ptr<f32>) {
  %ptrs = tt.splat %ptr : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>>
  %range = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %rows = tt.expand_dims %range {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
  %c16 = arith.constant 16 : i32
  %limits = tt.splat %c16 : i32 -> tensor<32x1xi32>
  %cmp = arith.cmpi slt, %rows, %limits : tensor<32x1xi32>
  %mask = tt.broadcast %cmp : tensor<32x1xi1> -> tensor<32x32xi1>
  %zero = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
  %a = tt.load %ptrs, %mask, %zero : tensor<32x32x!tt.ptr<f32>>
  %b = tt.load %ptrs : tensor<32x32x!tt.ptr<f32>>
  %empty = tensor.empty() : tensor<32x32xf32>
  %scalar_zero = arith.constant 0.0 : f32
  %acc = linalg.fill ins(%scalar_zero : f32) outs(%empty : tensor<32x32xf32>) -> tensor<32x32xf32>
  %result = linalg.matmul ins(%a, %b : tensor<32x32xf32>, tensor<32x32xf32>)
                          outs(%acc : tensor<32x32xf32>) -> tensor<32x32xf32>
  tt.return
}
