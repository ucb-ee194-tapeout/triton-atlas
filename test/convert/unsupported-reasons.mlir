// RUN: not triton-atlas-opt %s --split-input-file --convert-triton-to-atlas-virtual 2>&1 | FileCheck %s

// Pass 7 rejects what it cannot lower with the operation and the reason named, not a bare 'failed to legalize'.

// A generic with a broadcast operand (a non-identity indexing map): lowering it as pointwise would give a wrong tile.
// CHECK: atlas: unsupported linalg.generic: pass 7 lowers only pointwise generics
module {
  tt.func @k(%x: !tt.ptr<bf16>, %out: !tt.ptr<bf16>) {
    %xp = tt.splat %x : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    %op = tt.splat %out : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    %input = tt.load %xp {ttatlas.slot = #ttatlas.slot<index = 0, arg = 0, role = bias_bf16, tile_origin = affine_map<()[s0, s1, s2, s3, s4, s5] -> (0, 0)>, logical_shape = [32, 32], host_dtype = bf16>} : tensor<32x32x!tt.ptr<bf16>>
    %v = tensor.empty() : tensor<32xbf16>
    %e = tensor.empty() : tensor<32x32xbf16>
    %result = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%input, %v : tensor<32x32xbf16>, tensor<32xbf16>) outs(%e : tensor<32x32xbf16>) {
    ^bb0(%a: bf16, %b: bf16, %o: bf16):
      %s = arith.addf %a, %b : bf16
      linalg.yield %s : bf16
    } -> tensor<32x32xbf16>
    tt.store %op, %result {ttatlas.slot = #ttatlas.slot<index = 0, arg = 1, role = output_bf16, tile_origin = affine_map<()[s0, s1, s2, s3, s4, s5] -> (0, 0)>, logical_shape = [32, 32], host_dtype = bf16>} : tensor<32x32x!tt.ptr<bf16>>
    tt.return
  }
}

// -----

// A broadcast of something that is neither a reduction's result nor a bias row is not already laid out across the tile.
// CHECK: atlas: unsupported linalg.broadcast: a broadcast is lowered only for a row reduction's result or a bias row
module {
  tt.func @k(%x: !tt.ptr<bf16>, %out: !tt.ptr<bf16>) {
    %xp = tt.splat %x : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    %op = tt.splat %out : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    %input = tt.load %xp {ttatlas.slot = #ttatlas.slot<index = 0, arg = 0, role = bias_bf16, tile_origin = affine_map<()[s0, s1, s2, s3, s4, s5] -> (0, 0)>, logical_shape = [32, 32], host_dtype = bf16>} : tensor<32x32x!tt.ptr<bf16>>
    %v = tensor.empty() : tensor<32xbf16>
    %e = tensor.empty() : tensor<32x32xbf16>
    %result = linalg.broadcast ins(%v : tensor<32xbf16>) outs(%e : tensor<32x32xbf16>) dimensions = [1]
    tt.store %op, %result {ttatlas.slot = #ttatlas.slot<index = 0, arg = 1, role = output_bf16, tile_origin = affine_map<()[s0, s1, s2, s3, s4, s5] -> (0, 0)>, logical_shape = [32, 32], host_dtype = bf16>} : tensor<32x32x!tt.ptr<bf16>>
    tt.return
  }
}

// -----

// A named linalg op (here a plain add) is not one of the pointwise generics pass 3 emits, and has no lowering yet.
// CHECK: atlas: unsupported linalg.add: pass 7 has no Atlas lowering for linalg.add yet
module {
  tt.func @k(%x: !tt.ptr<bf16>, %out: !tt.ptr<bf16>) {
    %xp = tt.splat %x : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    %op = tt.splat %out : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    %input = tt.load %xp {ttatlas.slot = #ttatlas.slot<index = 0, arg = 0, role = bias_bf16, tile_origin = affine_map<()[s0, s1, s2, s3, s4, s5] -> (0, 0)>, logical_shape = [32, 32], host_dtype = bf16>} : tensor<32x32x!tt.ptr<bf16>>
    %e = tensor.empty() : tensor<32x32xbf16>
    %result = linalg.add ins(%input, %input : tensor<32x32xbf16>, tensor<32x32xbf16>) outs(%e : tensor<32x32xbf16>) -> tensor<32x32xbf16>
    tt.store %op, %result {ttatlas.slot = #ttatlas.slot<index = 0, arg = 1, role = output_bf16, tile_origin = affine_map<()[s0, s1, s2, s3, s4, s5] -> (0, 0)>, logical_shape = [32, 32], host_dtype = bf16>} : tensor<32x32x!tt.ptr<bf16>>
    tt.return
  }
}
