// RUN: triton-atlas-opt %s --convert-triton-to-atlas-virtual --verify-atlas-virtual-stream | FileCheck %s

// ttatlas.kernel (schema v2) is built from the slots alone: every field of a memory slot, a constant slot's raw pattern,
// and the DRAM bases (the module's ttatlas.dram_base, then the output window right after the inputs). No slot or other
// ttatlas attribute is left on an op atlas-mlir parses.

// CHECK-LABEL: func.func @metadata()
// CHECK-SAME: atlas.input_dram_base = 2432696320 : i64
// CHECK-SAME: atlas.output_dram_base = 2432700416 : i64
// CHECK-SAME: \22dram_base\22:{\22input\22:2432696320,\22output\22:2432700416}
// CHECK-SAME: \22grid\22:[1,1,1]
// CHECK-SAME: \22inputs\22:[{\22block\22:[0,0,32,32],\22host_dtype\22:\22bf16\22,\22index\22:0,\22kind\22:\22constant\22,\22pattern\22:[32767,32767,
// CHECK-SAME: {\22arg\22:0,\22block\22:[0,0,32,16],\22col_stride\22:2,\22host_dtype\22:\22bf16\22,\22index\22:1,\22kind\22:\22memory\22,\22offset\22:96,\22other\22:65408,\22role\22:\22constant_bf16\22,\22row_stride\22:64,
// CHECK-SAME: \22transposed\22:true
// CHECK-SAME: \22name\22:\22metadata\22
// CHECK-SAME: \22outputs\22:[{\22arg\22:1
// CHECK-SAME: \22version\22:2
// CHECK-NOT: ttatlas.slot
#constant = #ttatlas.slot<index = 0, kind = constant, role = constant_bf16, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], pattern = dense<32767> : tensor<32x32xi16>, vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
#memory = #ttatlas.slot<index = 1, kind = memory, arg = 0, role = constant_bf16, offset = 96, row_stride = 64, col_stride = 2, block = [0, 0, 32, 16], valid = [0, 0, 32, 16], other = -inf : bf16, vector_layout = none, transposed = true, host_dtype = bf16, tile_dtype = bf16>
#out0 = #ttatlas.slot<index = 0, kind = memory, arg = 1, role = output_bf16, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
#out1 = #ttatlas.slot<index = 1, kind = memory, arg = 1, role = output_bf16, offset = 1024, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = bf16, tile_dtype = bf16>
module attributes {ttatlas.dram_base = 2432696320 : i64} {
  tt.func @metadata(%a: !tt.ptr<bf16>, %b: !tt.ptr<bf16>) {
    %c = arith.constant {ttatlas.slot = #constant} dense<0.000000e+00> : tensor<32x32xbf16>
    %p = tt.splat %a : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    %q = tt.splat %b : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    %v = tt.load %p {ttatlas.slot = #memory} : tensor<32x32x!tt.ptr<bf16>>
    tt.store %q, %c {ttatlas.slot = #out0} : tensor<32x32x!tt.ptr<bf16>>
    tt.store %q, %v {ttatlas.slot = #out1} : tensor<32x32x!tt.ptr<bf16>>
    tt.return
  }
}
