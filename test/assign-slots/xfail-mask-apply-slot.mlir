// RUN: triton-atlas-opt %s --ttatlas-assign-slots | FileCheck %s
// XFAIL: *

// Planned for the next pass 6 PR (mask slots from ttatlas.mask_apply constants). A ttatlas.mask_apply constant is a tile the host fills, so pass 6 should give it its own mask_bf16 slot
// (and only a mask_apply constant makes one: a pad value never does). Today the constant stays an in-kernel constant
// with no slot.
// CHECK: role = mask_bf16
module {
  tt.func @masked_max(%a: !tt.ptr<bf16>, %o: !tt.ptr<bf16>) {
    %r = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
    %rows = tt.expand_dims %r {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %cols = tt.expand_dims %r {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %s = arith.constant dense<32> : tensor<32x1xi32>
    %scaled = arith.muli %rows, %s : tensor<32x1xi32>
    %rr = tt.broadcast %scaled : tensor<32x1xi32> -> tensor<32x32xi32>
    %cc = tt.broadcast %cols : tensor<1x32xi32> -> tensor<32x32xi32>
    %offsets = arith.addi %rr, %cc : tensor<32x32xi32>
    %base = tt.splat %a : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    %p = tt.addptr %base, %offsets : tensor<32x32x!tt.ptr<bf16>>, tensor<32x32xi32>
    %obase = tt.splat %o : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    %op = tt.addptr %obase, %offsets : tensor<32x32x!tt.ptr<bf16>>, tensor<32x32xi32>
    %x = tt.load %p {ttatlas.host_dtype = bf16} : tensor<32x32x!tt.ptr<bf16>>
    %mask = arith.constant dense<0.000000e+00> : tensor<32x32xbf16>
    %e = tensor.empty() : tensor<32x32xbf16>
    %y = linalg.add {ttatlas.mask_apply} ins(%x, %mask : tensor<32x32xbf16>, tensor<32x32xbf16>) outs(%e : tensor<32x32xbf16>) -> tensor<32x32xbf16>
    tt.store %op, %y {ttatlas.host_dtype = bf16} : tensor<32x32x!tt.ptr<bf16>>
    tt.return
  }
}
