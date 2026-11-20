// RUN: triton-atlas-opt %s --split-input-file --verify-diagnostics --ttatlas-assign-slots

// Pass 6 rejects what it cannot describe as a slot, by name. Copies (a tt.load feeding a tt.store, one tile) keep the
// cases short.

// Pointer that is not an argument plus compile-time offsets.
tt.func @dynamic_pointer(%p: !tt.ptr<bf16>, %q: !tt.ptr<bf16>, %n: i32) {
  %r = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
  %nn = tt.splat %n : i32 -> tensor<32xi32>
  %o = arith.addi %r, %nn : tensor<32xi32>
  %s = tt.splat %p : !tt.ptr<bf16> -> tensor<32x!tt.ptr<bf16>>
  %a = tt.addptr %s, %o : tensor<32x!tt.ptr<bf16>>, tensor<32xi32>
  // expected-error @below {{cannot prove a pointer that is a kernel argument plus compile-time offsets}}
  %v = tt.load %a : tensor<32x!tt.ptr<bf16>>
  %sq = tt.splat %q : !tt.ptr<bf16> -> tensor<32x!tt.ptr<bf16>>
  %b = tt.addptr %sq, %r : tensor<32x!tt.ptr<bf16>>, tensor<32xi32>
  tt.store %b, %v : tensor<32x!tt.ptr<bf16>>
  tt.return
}

// -----

// A stride that changes along a row: offsets r * c.
tt.func @non_affine(%p: !tt.ptr<bf16>, %q: !tt.ptr<bf16>) {
  %r = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
  %sq = arith.muli %r, %r : tensor<32xi32>
  %s = tt.splat %p : !tt.ptr<bf16> -> tensor<32x!tt.ptr<bf16>>
  %a = tt.addptr %s, %sq : tensor<32x!tt.ptr<bf16>>, tensor<32xi32>
  // expected-error @below {{a non-constant stride}}
  %v = tt.load %a : tensor<32x!tt.ptr<bf16>>
  %sq2 = tt.splat %q : !tt.ptr<bf16> -> tensor<32x!tt.ptr<bf16>>
  %b = tt.addptr %sq2, %r : tensor<32x!tt.ptr<bf16>>, tensor<32xi32>
  tt.store %b, %v : tensor<32x!tt.ptr<bf16>>
  tt.return
}

// -----

// A mask whose true lanes are not one rectangle: (row + col) < 32 is a triangle.
tt.func @triangle_mask(%p: !tt.ptr<bf16>, %q: !tt.ptr<bf16>) {
  %r = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
  %rows = tt.expand_dims %r {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
  %cols = tt.expand_dims %r {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
  %rb = tt.broadcast %rows : tensor<32x1xi32> -> tensor<32x32xi32>
  %cb = tt.broadcast %cols : tensor<1x32xi32> -> tensor<32x32xi32>
  %sum = arith.addi %rb, %cb : tensor<32x32xi32>
  %lim = arith.constant dense<32> : tensor<32x32xi32>
  %mask = arith.cmpi slt, %sum, %lim : tensor<32x32xi32>
  %c32 = arith.constant dense<32> : tensor<32x1xi32>
  %rowoff = arith.muli %rows, %c32 : tensor<32x1xi32>
  %rowoffb = tt.broadcast %rowoff : tensor<32x1xi32> -> tensor<32x32xi32>
  %off = arith.addi %rowoffb, %cb : tensor<32x32xi32>
  %s = tt.splat %p : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
  %a = tt.addptr %s, %off : tensor<32x32x!tt.ptr<bf16>>, tensor<32x32xi32>
  // expected-error @below {{a mask that is not one compile-time rectangle}}
  %v = tt.load %a, %mask : tensor<32x32x!tt.ptr<bf16>>
  %sq = tt.splat %q : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
  %b = tt.addptr %sq, %off : tensor<32x32x!tt.ptr<bf16>>, tensor<32x32xi32>
  tt.store %b, %v : tensor<32x32x!tt.ptr<bf16>>
  tt.return
}

// -----

// Two stores to the same elements.
tt.func @overlapping_stores(%p: !tt.ptr<bf16>, %q: !tt.ptr<bf16>) {
  %r = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
  %s = tt.splat %p : !tt.ptr<bf16> -> tensor<32x!tt.ptr<bf16>>
  %a = tt.addptr %s, %r : tensor<32x!tt.ptr<bf16>>, tensor<32xi32>
  %v = tt.load %a : tensor<32x!tt.ptr<bf16>>
  %sq = tt.splat %q : !tt.ptr<bf16> -> tensor<32x!tt.ptr<bf16>>
  %b = tt.addptr %sq, %r : tensor<32x!tt.ptr<bf16>>, tensor<32xi32>
  tt.store %b, %v : tensor<32x!tt.ptr<bf16>>
  // expected-error @below {{overlapping store footprints}}
  tt.store %b, %v : tensor<32x!tt.ptr<bf16>>
  tt.return
}

// -----

// A load that reads what an earlier store wrote.
tt.func @load_after_store(%p: !tt.ptr<bf16>, %q: !tt.ptr<bf16>) {
  %r = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
  %s = tt.splat %p : !tt.ptr<bf16> -> tensor<32x!tt.ptr<bf16>>
  %a = tt.addptr %s, %r : tensor<32x!tt.ptr<bf16>>, tensor<32xi32>
  %v = tt.load %a : tensor<32x!tt.ptr<bf16>>
  tt.store %a, %v : tensor<32x!tt.ptr<bf16>>
  // expected-error @below {{a load reads what an earlier store wrote}}
  %w = tt.load %a : tensor<32x!tt.ptr<bf16>>
  %sq = tt.splat %q : !tt.ptr<bf16> -> tensor<32x!tt.ptr<bf16>>
  %b = tt.addptr %sq, %r : tensor<32x!tt.ptr<bf16>>, tensor<32xi32>
  tt.store %b, %w : tensor<32x!tt.ptr<bf16>>
  tt.return
}

// -----

// A store whose own lanes overlap: every row writes the same 32 elements (row stride 0).
tt.func @store_row_stride_zero(%p: !tt.ptr<bf16>, %q: !tt.ptr<bf16>) {
  %r = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
  %c32 = arith.constant dense<32> : tensor<32x1xi32>
  %rows = tt.expand_dims %r {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
  %rowoff = arith.muli %rows, %c32 : tensor<32x1xi32>
  %cols = tt.expand_dims %r {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
  %rowb = tt.broadcast %rowoff : tensor<32x1xi32> -> tensor<32x32xi32>
  %colb = tt.broadcast %cols : tensor<1x32xi32> -> tensor<32x32xi32>
  %off = arith.addi %rowb, %colb : tensor<32x32xi32>
  %s = tt.splat %p : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
  %a = tt.addptr %s, %off : tensor<32x32x!tt.ptr<bf16>>, tensor<32x32xi32>
  %v = tt.load %a : tensor<32x32x!tt.ptr<bf16>>
  %sq = tt.splat %q : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
  %b = tt.addptr %sq, %colb : tensor<32x32x!tt.ptr<bf16>>, tensor<32x32xi32>
  // expected-error @below {{overlapping store footprints}}
  tt.store %b, %v : tensor<32x32x!tt.ptr<bf16>>
  tt.return
}

// -----

// An FP8 slot cannot hold a non-finite other: 0x7F is NaN in E4M3FN.
tt.func @fp8_nan_other(%p: !tt.ptr<f8E4M3FN>, %q: !tt.ptr<f8E4M3FN>) {
  %r = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
  %c32 = arith.constant dense<32> : tensor<32x1xi32>
  %rows = tt.expand_dims %r {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
  %rowoff = arith.muli %rows, %c32 : tensor<32x1xi32>
  %cols = tt.expand_dims %r {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
  %rowb = tt.broadcast %rowoff : tensor<32x1xi32> -> tensor<32x32xi32>
  %colb = tt.broadcast %cols : tensor<1x32xi32> -> tensor<32x32xi32>
  %off = arith.addi %rowb, %colb : tensor<32x32xi32>
  %lim = arith.constant dense<20> : tensor<1x32xi32>
  %m1 = arith.cmpi slt, %cols, %lim : tensor<1x32xi32>
  %mask = tt.broadcast %m1 : tensor<1x32xi1> -> tensor<32x32xi1>
  %other = arith.constant dense<0x7F> : tensor<32x32xf8E4M3FN>
  %s = tt.splat %p : !tt.ptr<f8E4M3FN> -> tensor<32x32x!tt.ptr<f8E4M3FN>>
  %a = tt.addptr %s, %off : tensor<32x32x!tt.ptr<f8E4M3FN>>, tensor<32x32xi32>
  // expected-error @below {{an FP8 slot cannot hold a non-finite other}}
  %v = tt.load %a, %mask, %other : tensor<32x32x!tt.ptr<f8E4M3FN>>
  %sq = tt.splat %q : !tt.ptr<f8E4M3FN> -> tensor<32x32x!tt.ptr<f8E4M3FN>>
  %b = tt.addptr %sq, %off : tensor<32x32x!tt.ptr<f8E4M3FN>>, tensor<32x32xi32>
  tt.store %b, %v : tensor<32x32x!tt.ptr<f8E4M3FN>>
  tt.return
}

// -----

// A program id in the address is not supported yet (offsets are plain integers).
tt.func @pid_in_address(%p: !tt.ptr<bf16>, %q: !tt.ptr<bf16>) {
  %pid = tt.get_program_id x : i32
  %r = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
  %pp = tt.splat %pid : i32 -> tensor<32xi32>
  %o = arith.addi %r, %pp : tensor<32xi32>
  %s = tt.splat %p : !tt.ptr<bf16> -> tensor<32x!tt.ptr<bf16>>
  %a = tt.addptr %s, %o : tensor<32x!tt.ptr<bf16>>, tensor<32xi32>
  // expected-error @below {{the pointer depends on a program id}}
  %v = tt.load %a : tensor<32x!tt.ptr<bf16>>
  %sq = tt.splat %q : !tt.ptr<bf16> -> tensor<32x!tt.ptr<bf16>>
  %b = tt.addptr %sq, %r : tensor<32x!tt.ptr<bf16>>, tensor<32xi32>
  tt.store %b, %v : tensor<32x!tt.ptr<bf16>>
  tt.return
}

// -----

// A negative stride (reversed row): the packer reads forward from the argument.
tt.func @negative_stride(%p: !tt.ptr<bf16>, %q: !tt.ptr<bf16>) {
  %r = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
  %top = arith.constant dense<31> : tensor<32xi32>
  %rev = arith.subi %top, %r : tensor<32xi32>
  %s = tt.splat %p : !tt.ptr<bf16> -> tensor<32x!tt.ptr<bf16>>
  %a = tt.addptr %s, %rev : tensor<32x!tt.ptr<bf16>>, tensor<32xi32>
  // expected-error @below {{a negative stride}}
  %v = tt.load %a : tensor<32x!tt.ptr<bf16>>
  %sq = tt.splat %q : !tt.ptr<bf16> -> tensor<32x!tt.ptr<bf16>>
  %b = tt.addptr %sq, %r : tensor<32x!tt.ptr<bf16>>, tensor<32xi32>
  tt.store %b, %v : tensor<32x!tt.ptr<bf16>>
  tt.return
}
