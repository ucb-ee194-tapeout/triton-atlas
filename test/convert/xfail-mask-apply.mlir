// RUN: triton-atlas-opt %s --convert-triton-to-atlas-virtual --verify-atlas-virtual-stream | FileCheck %s
// XFAIL: *

// A ttatlas.mask_apply, as pass 4 leaves it before a reduction that reads padded lanes: an add with a constant mask
// tile. Pass 6 does not give that tile a slot yet (a planned change of the slot-assignment lane), so pass 7 rejects it
// ("a ttatlas.mask_apply needs its mask tile as a slot"). Once it does, the add should lower to a VPU add of the loaded
// tile and the mask slot's tile.
// CHECK: atlas.virtual_vpu_binary
// CHECK-SAME: kind = "add"
module {
  tt.func @k(%x: !tt.ptr<bf16>, %out: !tt.ptr<bf16>) {
    %xp = tt.splat %x : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    %op = tt.splat %out : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    %input = tt.load %xp {ttatlas.slot = #ttatlas.slot<index = 0, arg = 0, role = bias_bf16, tile_origin = affine_map<()[s0, s1, s2, s3, s4, s5] -> (0, 0)>, logical_shape = [32, 32], host_dtype = bf16>} : tensor<32x32x!tt.ptr<bf16>>
    %mask = arith.constant dense<0.000000e+00> : tensor<32x32xbf16>
    %e = tensor.empty() : tensor<32x32xbf16>
    %result = linalg.add {ttatlas.mask_apply} ins(%input, %mask : tensor<32x32xbf16>, tensor<32x32xbf16>) outs(%e : tensor<32x32xbf16>) -> tensor<32x32xbf16>
    tt.store %op, %result {ttatlas.slot = #ttatlas.slot<index = 0, arg = 1, role = output_bf16, tile_origin = affine_map<()[s0, s1, s2, s3, s4, s5] -> (0, 0)>, logical_shape = [32, 32], host_dtype = bf16>} : tensor<32x32x!tt.ptr<bf16>>
    tt.return
  }
}
