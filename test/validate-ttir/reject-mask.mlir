// RUN: triton-atlas-opt %s --split-input-file --ttatlas-validate-ttir --verify-diagnostics

// Masks (arith.cmpi, arith.andi): pass 4 doesn't turn them into padding yet. When it does, these rows become legal in
// the same change, and pass 4 owns the rules on masks (runtime bounds, the load's `other`). Any mask operand on a
// load or store is rejected too, whatever builds it.

// A tail mask, offs < 20, as tl.load(..., mask=offs < 20) builds it.
tt.func @tail(%p: !tt.ptr<bf16>) {
  %c20 = arith.constant dense<20> : tensor<32xi32>
  %r = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  // expected-error @below {{atlas: unsupported arith.cmpi: no lowering yet: masks; pass 4 doesn't turn them into padding yet}}
  // expected-error @below {{failed to legalize operation 'arith.cmpi' that was explicitly marked illegal}}
  %m = arith.cmpi slt, %r, %c20 : tensor<32xi32>
  %ps = tt.splat %p : !tt.ptr<bf16> -> tensor<32x!tt.ptr<bf16>>
  %a = tt.addptr %ps, %r : tensor<32x!tt.ptr<bf16>>, tensor<32xi32>
  %x = tt.load %a, %m : tensor<32x!tt.ptr<bf16>>
  tt.return
}

// -----

// Two masks combined, as a 2-D tile's row and column masks are.
tt.func @both(%rows: tensor<32x32xi1>, %cols: tensor<32x32xi1>) {
  // expected-error @below {{atlas: unsupported arith.andi: no lowering yet: masks; pass 4 doesn't turn them into padding yet}}
  // expected-error @below {{failed to legalize operation 'arith.andi' that was explicitly marked illegal}}
  %m = arith.andi %rows, %cols : tensor<32x32xi1>
  tt.return
}

// -----

// A mask that isn't built from cmpi (here a constant) is rejected on the load.
tt.func @constant_mask(%p: !tt.ptr<bf16>) {
  %m = arith.constant dense<[true, false, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true]> : tensor<32xi1>
  %r = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %ps = tt.splat %p : !tt.ptr<bf16> -> tensor<32x!tt.ptr<bf16>>
  %a = tt.addptr %ps, %r : tensor<32x!tt.ptr<bf16>>, tensor<32xi32>
  // expected-error @below {{atlas: unsupported tt.load: no lowering yet: masks; pass 4 doesn't turn them into padding yet}}
  // expected-error @below {{failed to legalize operation 'tt.load' that was explicitly marked illegal}}
  %x = tt.load %a, %m : tensor<32x!tt.ptr<bf16>>
  tt.return
}

// -----

tt.func @masked_store(%p: !tt.ptr<bf16>, %x: tensor<32xbf16>) {
  %m = arith.constant dense<true> : tensor<32xi1>
  %r = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %ps = tt.splat %p : !tt.ptr<bf16> -> tensor<32x!tt.ptr<bf16>>
  %a = tt.addptr %ps, %r : tensor<32x!tt.ptr<bf16>>, tensor<32xi32>
  // expected-error @below {{atlas: unsupported tt.store: no lowering yet: masks; pass 4 doesn't turn them into padding yet}}
  // expected-error @below {{failed to legalize operation 'tt.store' that was explicitly marked illegal}}
  tt.store %a, %x, %m : tensor<32x!tt.ptr<bf16>>
  tt.return
}
