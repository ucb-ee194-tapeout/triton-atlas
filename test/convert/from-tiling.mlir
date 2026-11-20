// RUN: triton-atlas-opt %S/../inputs/stage3/copy_rows_50.mlir --ttatlas-legalize-dtypes --canonicalize --cse --ttatlas-legalize-shapes --ttatlas-assign-slots --convert-triton-to-atlas-virtual --verify-atlas-virtual-stream | FileCheck %s --check-prefix=COPY --implicit-check-not=tt.
// RUN: triton-atlas-opt %S/../inputs/stage3/softmax_16.mlir --ttatlas-legalize-dtypes --canonicalize --cse --ttatlas-legalize-shapes --ttatlas-assign-slots --convert-triton-to-atlas-virtual --verify-atlas-virtual-stream | FileCheck %s --check-prefix=SOFTMAX --implicit-check-not=tt.

// Pass 7 on what passes 5, 4 and 6 emit for real kernels, with the Atlas verifier run on the result.

// A 64x32 block copied with no compute: the load and the store carry one slot per tile, so each tile is read, moved
// and written in turn.
// COPY: atlas.virtual_input_bf16
// COPY-SAME: index = 0
// COPY: atlas.virtual_vpu_unary
// COPY-SAME: kind = "mov"
// COPY: atlas.virtual_output_bf16
// COPY-SAME: index = 0
// COPY: atlas.virtual_input_bf16
// COPY-SAME: index = 1
// COPY: atlas.virtual_output_bf16
// COPY-SAME: index = 1

// A softmax of a 16x16 block. The max reduction reads x through the clamp min(x, hi) (one constant tile, input 1); the
// sum reads exp(x - max) through min then max (two constant tiles, inputs 2 and 3). The exp input is clamped to -200.
// SOFTMAX: kind = "min"
// SOFTMAX: kind = "row_max"
// SOFTMAX: atlas.virtual_vpu_splat
// SOFTMAX-SAME: bits = 49992
// SOFTMAX: kind = "exp"
// SOFTMAX: atlas.virtual_input_bf16
// SOFTMAX-SAME: index = 2
// SOFTMAX: kind = "min"
// SOFTMAX: atlas.virtual_input_bf16
// SOFTMAX-SAME: index = 3
// SOFTMAX: kind = "max"
// SOFTMAX: kind = "row_sum"
// SOFTMAX: kind = "recip"
// SOFTMAX: kind = "mul"
// SOFTMAX: atlas.virtual_output_bf16
