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
  // expected-error @below {{edge mask is not a compile-time leading box}}
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
// turns each padded lane into 1, and the row sum then counts them. Pass 4 re-masks before the sum, so it does not:
// not tag the load as padded with zero. Only kernel inputs get their padded lanes filled, so the row sum reads a mask
// apply: x * M, with M = 1 on the 16 real columns and 0 on the padded ones.
#id = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: tt.func @padded_lanes_change_before_sum
// CHECK: tt.load {{.*}}ttatlas.pad = #ttatlas.pad<neutral = zero, logical_shape = [32, 16]>
// CHECK: linalg.generic
// CHECK: %[[MASKED:.*]] = linalg.mul {ttatlas.mask_apply} ins(%{{.*}}, %{{.*}} : tensor<32x32xf32>, tensor<32x32xf32>)
// CHECK: linalg.reduce ins(%[[MASKED]] : tensor<32x32xf32>)
tt.func @padded_lanes_change_before_sum(%ptr: !tt.ptr<f32>) -> tensor<32xf32> {
  %ptrs = tt.splat %ptr : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>>
  %range = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %limit = arith.constant 16 : i32
  %limits = tt.splat %limit : i32 -> tensor<32xi32>
  %cmp = arith.cmpi slt, %range, %limits : tensor<32xi32>
  %col_mask = tt.expand_dims %cmp {axis = 0 : i32} : tensor<32xi1> -> tensor<1x32xi1>
  %mask = tt.broadcast %col_mask : tensor<1x32xi1> -> tensor<32x32xi1>
  %zero = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
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
  tt.store %ptrs, %result : tensor<32x32x!tt.ptr<f32>>
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
  tt.store %ptrs, %result : tensor<32x32x!tt.ptr<f32>>
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
  tt.store %ptrs, %result : tensor<32x32x!tt.ptr<f32>>
  tt.return
}

// -----

// A 64-row block masked to its first 50 rows: two row tiles. Tile 0 is all real lanes, so it has no tag; tile 1 holds
// 18 real rows, so its load slice and its stored result each carry [18, 32], the tile's own shape, not the 50x32 of the
// whole block (which tiling would otherwise copy onto both tiles).
// CHECK-LABEL: tt.func @two_row_tiles
// CHECK: tensor.extract_slice {{.*}} [32, 32] [1, 1] {ttatlas.pad = #ttatlas.pad<neutral = zero, logical_shape = [18, 32]>}
// CHECK: linalg.add {ttatlas.pad = #ttatlas.pad<neutral = zero, logical_shape = [18, 32]>}
// CHECK-NOT: logical_shape = [50, 32]
tt.func @two_row_tiles(%ptr: !tt.ptr<bf16>) {
  %ptrs = tt.splat %ptr : !tt.ptr<bf16> -> tensor<64x32x!tt.ptr<bf16>>
  %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
  %limit = arith.constant 50 : i32
  %limits = tt.splat %limit : i32 -> tensor<64xi32>
  %cmp = arith.cmpi slt, %range, %limits : tensor<64xi32>
  %rows = tt.expand_dims %cmp {axis = 1 : i32} : tensor<64xi1> -> tensor<64x1xi1>
  %mask = tt.broadcast %rows : tensor<64x1xi1> -> tensor<64x32xi1>
  %zero = arith.constant dense<0.000000e+00> : tensor<64x32xbf16>
  %x = tt.load %ptrs, %mask, %zero : tensor<64x32x!tt.ptr<bf16>>
  %e = tensor.empty() : tensor<64x32xbf16>
  %y = linalg.add ins(%x, %x : tensor<64x32xbf16>, tensor<64x32xbf16>) outs(%e : tensor<64x32xbf16>) -> tensor<64x32xbf16>
  tt.store %ptrs, %y, %mask : tensor<64x32x!tt.ptr<bf16>>
  tt.return
}

// -----

// A 64-row block masked to 20 rows: tile 1 has no real lane, but it is still read and written, so there is nothing to
// tell the host about it. Rejected, with a hint.
tt.func @empty_second_tile(%ptr: !tt.ptr<bf16>) {
  %ptrs = tt.splat %ptr : !tt.ptr<bf16> -> tensor<64x32x!tt.ptr<bf16>>
  %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
  %limit = arith.constant 20 : i32
  %limits = tt.splat %limit : i32 -> tensor<64xi32>
  %cmp = arith.cmpi slt, %range, %limits : tensor<64xi32>
  %rows = tt.expand_dims %cmp {axis = 1 : i32} : tensor<64xi1> -> tensor<64x1xi1>
  %mask = tt.broadcast %rows : tensor<64x1xi1> -> tensor<64x32xi1>
  %zero = arith.constant dense<0.000000e+00> : tensor<64x32xbf16>
  // expected-error @below {{the mask leaves a whole tile without a real lane}}
  %x = tt.load %ptrs, %mask, %zero : tensor<64x32x!tt.ptr<bf16>>
  %e = tensor.empty() : tensor<64x32xbf16>
  %y = linalg.add ins(%x, %x : tensor<64x32xbf16>, tensor<64x32xbf16>) outs(%e : tensor<64x32xbf16>) -> tensor<64x32xbf16>
  tt.store %ptrs, %y, %mask : tensor<64x32x!tt.ptr<bf16>>
  tt.return
}

// -----

// A conversion between a 16-column load and a row max: the max's own pad tag says -inf, but the host fills the load
// from the load's tag, and nothing promises the lanes still hold -inf after the conversion. Trusting the max's tag
// would leave an all-negative row with a max of 0. So the max reads a mask apply (x + M, M = 0 on real lanes, -inf on
// padded ones).
#id = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: tt.func @conversion_before_max
// CHECK: %[[MASKED:.*]] = linalg.add {ttatlas.mask_apply} ins(%{{.*}}, %{{.*}} : tensor<32x32xf32>, tensor<32x32xf32>)
// CHECK: linalg.reduce ins(%[[MASKED]] : tensor<32x32xf32>)
tt.func @conversion_before_max(%ptr: !tt.ptr<bf16>, %out: !tt.ptr<f32>) {
  %ptrs = tt.splat %ptr : !tt.ptr<bf16> -> tensor<32x16x!tt.ptr<bf16>>
  %x = tt.load %ptrs : tensor<32x16x!tt.ptr<bf16>>
  %e = tensor.empty() : tensor<32x16xf32>
  %wide = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]}
      ins(%x : tensor<32x16xbf16>) outs(%e : tensor<32x16xf32>) {
  ^bb0(%in: bf16, %o: f32):
    %c = arith.extf %in : bf16 to f32
    linalg.yield %c : f32
  } -> tensor<32x16xf32>
  %ninf = arith.constant 0xFF800000 : f32
  %e1 = tensor.empty() : tensor<32xf32>
  %init = linalg.fill ins(%ninf : f32) outs(%e1 : tensor<32xf32>) -> tensor<32xf32>
  %max = linalg.reduce ins(%wide : tensor<32x16xf32>) outs(%init : tensor<32xf32>) dimensions = [1]
  (%a: f32, %b: f32) {
    %m = arith.maxnumf %a, %b : f32
    linalg.yield %m : f32
  }
  %optrs = tt.splat %out : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
  tt.store %optrs, %max : tensor<32x!tt.ptr<f32>>
  tt.return
}

// -----

// Masked key columns, scores = q @ k, then a row max (attention): the scores' padded columns come out of the matmul
// as 0, and a 0 in an all-negative row would win the max. The valid columns are followed through the matmul (the
// output's columns come from k's), so the max reads a mask apply that adds -inf to them.
// CHECK-LABEL: tt.func @masked_dot_then_row_max
// CHECK: linalg.matmul
// CHECK: %[[MASKED:.*]] = linalg.add {ttatlas.mask_apply} ins(%{{.*}}, %{{.*}} : tensor<32x32xf32>, tensor<32x32xf32>)
// CHECK: linalg.reduce ins(%[[MASKED]] : tensor<32x32xf32>)
tt.func @masked_dot_then_row_max(%qp: !tt.ptr<f32>, %kp: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
  %qptrs = tt.splat %qp : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>>
  %kptrs = tt.splat %kp : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>>
  %range = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %limit = arith.constant 20 : i32
  %limits = tt.splat %limit : i32 -> tensor<32xi32>
  %cmp = arith.cmpi slt, %range, %limits : tensor<32xi32>
  %col_mask = tt.expand_dims %cmp {axis = 0 : i32} : tensor<32xi1> -> tensor<1x32xi1>
  %mask = tt.broadcast %col_mask : tensor<1x32xi1> -> tensor<32x32xi1>
  %zero = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
  %q = tt.load %qptrs : tensor<32x32x!tt.ptr<f32>>
  %k = tt.load %kptrs, %mask, %zero : tensor<32x32x!tt.ptr<f32>>
  %e = tensor.empty() : tensor<32x32xf32>
  %z = arith.constant 0.0 : f32
  %acc = linalg.fill ins(%z : f32) outs(%e : tensor<32x32xf32>) -> tensor<32x32xf32>
  %s = linalg.matmul ins(%q, %k : tensor<32x32xf32>, tensor<32x32xf32>)
                     outs(%acc : tensor<32x32xf32>) -> tensor<32x32xf32>
  %ninf = arith.constant 0xFF800000 : f32
  %e1 = tensor.empty() : tensor<32xf32>
  %init = linalg.fill ins(%ninf : f32) outs(%e1 : tensor<32xf32>) -> tensor<32xf32>
  %max = linalg.reduce ins(%s : tensor<32x32xf32>) outs(%init : tensor<32xf32>) dimensions = [1]
  (%a: f32, %b: f32) {
    %m = arith.maxnumf %a, %b : f32
    linalg.yield %m : f32
  }
  %optrs = tt.splat %out : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
  tt.store %optrs, %max : tensor<32x!tt.ptr<f32>>
  tt.return
}

// -----

// One masked load read by a row sum and a row max (the max last): a load has one fill. The max wants -inf and the sum 0, so the host
// fills 0 (a finite lane, which a mask apply handles exactly), the max reads an add mask, and the sum needs none. With
// -inf, the sum's x * 0 mask would be NaN.
// CHECK-LABEL: tt.func @sum_and_max_of_one_load
// CHECK: tt.load {{.*}}ttatlas.pad = #ttatlas.pad<neutral = zero, logical_shape = [32, 16]>
// CHECK-NOT: linalg.mul {ttatlas.mask_apply}
// CHECK: linalg.add {ttatlas.mask_apply}
// CHECK-NOT: linalg.mul {ttatlas.mask_apply}
tt.func @sum_and_max_of_one_load(%ptr: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
  %ptrs = tt.splat %ptr : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>>
  %range = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %limit = arith.constant 16 : i32
  %limits = tt.splat %limit : i32 -> tensor<32xi32>
  %cmp = arith.cmpi slt, %range, %limits : tensor<32xi32>
  %col_mask = tt.expand_dims %cmp {axis = 0 : i32} : tensor<32xi1> -> tensor<1x32xi1>
  %mask = tt.broadcast %col_mask : tensor<1x32xi1> -> tensor<32x32xi1>
  %zero = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
  %x = tt.load %ptrs, %mask, %zero : tensor<32x32x!tt.ptr<f32>>
  %ninf = arith.constant 0xFF800000 : f32
  %z = arith.constant 0.0 : f32
  %e2 = tensor.empty() : tensor<32xf32>
  %sum_init = linalg.fill ins(%z : f32) outs(%e2 : tensor<32xf32>) -> tensor<32xf32>
  %sum = linalg.reduce ins(%x : tensor<32x32xf32>) outs(%sum_init : tensor<32xf32>) dimensions = [1]
  (%a: f32, %b: f32) {
    %t = arith.addf %a, %b : f32
    linalg.yield %t : f32
  }
  %e1 = tensor.empty() : tensor<32xf32>
  %max_init = linalg.fill ins(%ninf : f32) outs(%e1 : tensor<32xf32>) -> tensor<32xf32>
  %max = linalg.reduce ins(%x : tensor<32x32xf32>) outs(%max_init : tensor<32xf32>) dimensions = [1]
  (%a: f32, %b: f32) {
    %m = arith.maxnumf %a, %b : f32
    linalg.yield %m : f32
  }
  %total = arith.addf %max, %sum : tensor<32xf32>
  %optrs = tt.splat %out : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
  tt.store %optrs, %total : tensor<32x!tt.ptr<f32>>
  tt.return
}
