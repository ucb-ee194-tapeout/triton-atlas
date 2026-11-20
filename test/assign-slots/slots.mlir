// RUN: triton-atlas-opt %S/../inputs/stage3/matmul_relu_64x32.mlir --ttatlas-legalize-dtypes --canonicalize --cse --ttatlas-legalize-shapes --ttatlas-assign-slots -o %t.mm
// RUN: FileCheck %s --input-file=%t.mm --check-prefix=MM
// RUN: triton-atlas-opt %S/../inputs/stage3/mlp.mlir --ttatlas-legalize-dtypes --canonicalize --cse --ttatlas-legalize-shapes --ttatlas-assign-slots -o %t.mlp
// RUN: FileCheck %s --input-file=%t.mlp --check-prefix=MLP
// RUN: triton-atlas-opt %S/../inputs/stage3/softmax_masked_20.mlir --ttatlas-legalize-dtypes --canonicalize --cse --ttatlas-legalize-shapes --ttatlas-assign-slots -o %t.masked
// RUN: FileCheck %s --input-file=%t.masked --check-prefix=MASKED
// RUN: triton-atlas-opt %S/../inputs/stage3/add_50x40_of_64.mlir --ttatlas-legalize-dtypes --canonicalize --cse --ttatlas-legalize-shapes --ttatlas-assign-slots -o %t.edge
// RUN: FileCheck %s --input-file=%t.edge --check-prefix=EDGE
// RUN: triton-atlas-opt %S/../inputs/stage3/softmax_16.mlir --ttatlas-legalize-dtypes --canonicalize --cse --ttatlas-legalize-shapes --ttatlas-assign-slots -o %t.pad
// RUN: FileCheck %s --input-file=%t.pad --check-prefix=PAD
// RUN: triton-atlas-opt %S/../inputs/stage3/copy_16.mlir --ttatlas-legalize-dtypes --canonicalize --cse --ttatlas-legalize-shapes --ttatlas-assign-slots -o %t.copy
// RUN: FileCheck %s --input-file=%t.copy --check-prefix=COPY

// RUN: triton-atlas-opt %S/../inputs/stage3/copy_rows_50.mlir --ttatlas-legalize-dtypes --canonicalize --cse --ttatlas-legalize-shapes --ttatlas-assign-slots -o %t.copy2
// RUN: FileCheck %s --input-file=%t.copy2 --check-prefix=COPY2
// RUN: triton-atlas-opt %S/../inputs/stage3/bias_both_layouts.mlir --ttatlas-legalize-dtypes --canonicalize --cse --ttatlas-legalize-shapes --ttatlas-assign-slots -o %t.both
// RUN: FileCheck %s --input-file=%t.both --check-prefix=BOTH
// RUN: triton-atlas-opt %S/../inputs/stage3/strided_cols.mlir --ttatlas-legalize-dtypes --canonicalize --cse --ttatlas-legalize-shapes --ttatlas-assign-slots -o %t.strided
// RUN: FileCheck %s --input-file=%t.strided --check-prefix=STRIDED

// Slots on the tiles of a 64x32 matmul + ReLU (2x1 grid): the pack carries one slot per tile in row-major grid order
// (offsets 0 and 1024 = one tile down); the weight is a plain [k][n] host tile that the pack turns (transposed), so its
// role is kn; each output tile has its own slot on the unpack.
// MM-DAG: linalg.pack {{.*}}ttatlas.slot = [#ttatlas.slot<index = 0, kind = memory, arg = 0, role = activation_fp8, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false, host_dtype = f8E4M3FN, tile_dtype = f8E4M3FN>, #ttatlas.slot<index = 1, kind = memory, arg = 0, role = activation_fp8, offset = 1024,
// MM-DAG: linalg.pack {{.*}}ttatlas.slot = #ttatlas.slot<index = 2, kind = memory, arg = 1, role = weight_fp8_kn, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = true
// MM-DAG: linalg.unpack {{.*}}ttatlas.slot = [#ttatlas.slot<index = 0, kind = memory, arg = 2, role = output_bf16, offset = 0, {{.*}}#ttatlas.slot<index = 1, kind = memory, arg = 2, role = output_bf16, offset = 1024,

// x @ tl.trans(w): the weight's host tile is [n][k] (nk) and is not turned; the bias vector feeds a row broadcast along
// the columns, so it is a vector along columns, its lanes along dim 0 of block and valid with lane stride in row_stride.
// MLP-DAG: role = bias_bf16, offset = 0, row_stride = 1, col_stride = 1, block = [0, 0, 32, 1], valid = [0, 0, 32, 1], vector_layout = cols
// MLP-DAG: role = weight_fp8_nk, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false

// A masked load (tl.load with a mask of 20 columns and other = -inf) keeps its box and its other, and the store the box.
// MASKED-DAG: valid = [0, 0, 32, 20], other = -inf : bf16
// MASKED-DAG: role = output_bf16, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 20]

// A 64x64 block whose mask keeps 50 rows and 40 columns: the second tile column has 8 valid lanes; offsets use the row
// stride 64 (the second tile is 32 elements right, the third 32 rows down).
// EDGE-DAG: offset = 32, row_stride = 64, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 8]
// EDGE-DAG: offset = 2048, row_stride = 64

// A 16x16 block is a block of 16x16 inside the tile; the clamp masks of the padded reductions are constant slots whose
// pattern holds the raw bits.
// PAD-DAG: kind = constant, role = mask_bf16, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], pattern = dense<"0xFF7FFF7F
// PAD-DAG: role = constant_bf16, offset = 0, row_stride = 16, col_stride = 1, block = [0, 0, 16, 16], valid = [0, 0, 16, 16]

// A kernel with no compute has its slots on the tt.load and tt.store.
// COPY: tt.load {{.*}}ttatlas.slot = #ttatlas.slot<index = 0, kind = memory, arg = 0, role = constant_bf16
// COPY: tt.store {{.*}}ttatlas.slot = #ttatlas.slot<index = 0, kind = memory, arg = 1, role = output_bf16

// A compute-free copy of a 64x32 block with 50 rows kept: the load and the store carry one slot per tile, the second
// tile 1024 elements down with 18 valid rows. The load's other (0) fills the rest of that tile.
// COPY2: tt.load {{.*}}ttatlas.slot = [#ttatlas.slot<index = 0, kind = memory, arg = 0, role = constant_bf16, offset = 0, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], other = 0.000000e+00 : bf16, {{.*}}#ttatlas.slot<index = 1, kind = memory, arg = 0, role = constant_bf16, offset = 1024, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 18, 32], other = 0.000000e+00 : bf16
// COPY2: tt.store {{.*}}ttatlas.slot = [#ttatlas.slot<index = 0, kind = memory, arg = 1, role = output_bf16, offset = 0, {{.*}}#ttatlas.slot<index = 1, kind = memory, arg = 1, role = output_bf16, offset = 1024, row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 18, 32]

// A bias added along the rows and along the columns is a loaded vector needed in both layouts: the pack is cloned and
// the host writes it twice, with its own slot index for each layout.
// BOTH-DAG: index = 0, kind = memory, arg = 1, role = bias_bf16, offset = 0, row_stride = 1, col_stride = 1, block = [0, 0, 32, 1], valid = [0, 0, 32, 1], vector_layout = cols
// BOTH-DAG: index = 2, kind = memory, arg = 1, role = bias_bf16, offset = 0, row_stride = 1, col_stride = 1, block = [0, 0, 32, 1], valid = [0, 0, 32, 1], vector_layout = rows

// A tile read with row stride 64 and column stride 2 (every other element of every other row).
// STRIDED: tt.load {{.*}}ttatlas.slot = #ttatlas.slot<index = 0, kind = memory, arg = 0, role = constant_bf16, offset = 0, row_stride = 64, col_stride = 2, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, transposed = false
