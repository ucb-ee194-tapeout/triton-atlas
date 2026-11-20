// RUN: triton-atlas-opt %S/../inputs/stage3/add_16.mlir --ttatlas-legalize-dtypes --canonicalize --cse --ttatlas-legalize-shapes --ttatlas-assign-slots --ttatlas-verify-contract=stage=6 -o %t.add_16
// RUN: FileCheck %s --input-file=%t.add_16 --check-prefix=add_16
// RUN: triton-atlas-opt %S/../inputs/stage3/add_50x40_of_64.mlir --ttatlas-legalize-dtypes --canonicalize --cse --ttatlas-legalize-shapes --ttatlas-assign-slots --ttatlas-verify-contract=stage=6 -o %t.add_50x40_of_64
// RUN: FileCheck %s --input-file=%t.add_50x40_of_64 --check-prefix=add_50x40_of_64
// RUN: triton-atlas-opt %S/../inputs/stage3/add_64x16.mlir --ttatlas-legalize-dtypes --canonicalize --cse --ttatlas-legalize-shapes --ttatlas-assign-slots --ttatlas-verify-contract=stage=6 -o %t.add_64x16
// RUN: FileCheck %s --input-file=%t.add_64x16 --check-prefix=add_64x16
// RUN: triton-atlas-opt %S/../inputs/stage3/add_rows_50.mlir --ttatlas-legalize-dtypes --canonicalize --cse --ttatlas-legalize-shapes --ttatlas-assign-slots --ttatlas-verify-contract=stage=6 -o %t.add_rows_50
// RUN: FileCheck %s --input-file=%t.add_rows_50 --check-prefix=add_rows_50
// RUN: triton-atlas-opt %S/../inputs/stage3/bias_as_column.mlir --ttatlas-legalize-dtypes --canonicalize --cse --ttatlas-legalize-shapes --ttatlas-assign-slots --ttatlas-verify-contract=stage=6 -o %t.bias_as_column
// RUN: FileCheck %s --input-file=%t.bias_as_column --check-prefix=bias_as_column
// RUN: triton-atlas-opt %S/../inputs/stage3/bias_seeded_k_loop.mlir --ttatlas-legalize-dtypes --canonicalize --cse --ttatlas-legalize-shapes --ttatlas-assign-slots --ttatlas-verify-contract=stage=6 -o %t.bias_seeded_k_loop
// RUN: FileCheck %s --input-file=%t.bias_seeded_k_loop --check-prefix=bias_seeded_k_loop
// RUN: triton-atlas-opt %S/../inputs/stage3/copy_16.mlir --ttatlas-legalize-dtypes --canonicalize --cse --ttatlas-legalize-shapes --ttatlas-assign-slots --ttatlas-verify-contract=stage=6 -o %t.copy_16
// RUN: FileCheck %s --input-file=%t.copy_16 --check-prefix=copy_16
// RUN: triton-atlas-opt %S/../inputs/stage3/dot_plus_dot.mlir --ttatlas-legalize-dtypes --canonicalize --cse --ttatlas-legalize-shapes --ttatlas-assign-slots --ttatlas-verify-contract=stage=6 -o %t.dot_plus_dot
// RUN: FileCheck %s --input-file=%t.dot_plus_dot --check-prefix=dot_plus_dot
// RUN: triton-atlas-opt %S/../inputs/stage3/masked_rows_50_matmul.mlir --ttatlas-legalize-dtypes --canonicalize --cse --ttatlas-legalize-shapes --ttatlas-assign-slots --ttatlas-verify-contract=stage=6 -o %t.masked_rows_50_matmul
// RUN: FileCheck %s --input-file=%t.masked_rows_50_matmul --check-prefix=masked_rows_50_matmul
// RUN: triton-atlas-opt %S/../inputs/stage3/matmul_64.mlir --ttatlas-legalize-dtypes --canonicalize --cse --ttatlas-legalize-shapes --ttatlas-assign-slots --ttatlas-verify-contract=stage=6 -o %t.matmul_64
// RUN: FileCheck %s --input-file=%t.matmul_64 --check-prefix=matmul_64
// RUN: triton-atlas-opt %S/../inputs/stage3/matmul_relu.mlir --ttatlas-legalize-dtypes --canonicalize --cse --ttatlas-legalize-shapes --ttatlas-assign-slots --ttatlas-verify-contract=stage=6 -o %t.matmul_relu
// RUN: FileCheck %s --input-file=%t.matmul_relu --check-prefix=matmul_relu
// RUN: triton-atlas-opt %S/../inputs/stage3/matmul_relu_64x32.mlir --ttatlas-legalize-dtypes --canonicalize --cse --ttatlas-legalize-shapes --ttatlas-assign-slots --ttatlas-verify-contract=stage=6 -o %t.matmul_relu_64x32
// RUN: FileCheck %s --input-file=%t.matmul_relu_64x32 --check-prefix=matmul_relu_64x32
// RUN: triton-atlas-opt %S/../inputs/stage3/mlp.mlir --ttatlas-legalize-dtypes --canonicalize --cse --ttatlas-legalize-shapes --ttatlas-assign-slots --ttatlas-verify-contract=stage=6 -o %t.mlp
// RUN: FileCheck %s --input-file=%t.mlp --check-prefix=mlp
// RUN: triton-atlas-opt %S/../inputs/stage3/mm32.mlir --ttatlas-legalize-dtypes --canonicalize --cse --ttatlas-legalize-shapes --ttatlas-assign-slots --ttatlas-verify-contract=stage=6 -o %t.mm32
// RUN: FileCheck %s --input-file=%t.mm32 --check-prefix=mm32
// RUN: triton-atlas-opt %S/../inputs/stage3/row_max_16.mlir --ttatlas-legalize-dtypes --canonicalize --cse --ttatlas-legalize-shapes --ttatlas-assign-slots --ttatlas-verify-contract=stage=6 -o %t.row_max_16
// RUN: FileCheck %s --input-file=%t.row_max_16 --check-prefix=row_max_16
// RUN: triton-atlas-opt %S/../inputs/stage3/row_max_64.mlir --ttatlas-legalize-dtypes --canonicalize --cse --ttatlas-legalize-shapes --ttatlas-assign-slots --ttatlas-verify-contract=stage=6 -o %t.row_max_64
// RUN: FileCheck %s --input-file=%t.row_max_64 --check-prefix=row_max_64
// RUN: triton-atlas-opt %S/../inputs/stage3/row_min_16.mlir --ttatlas-legalize-dtypes --canonicalize --cse --ttatlas-legalize-shapes --ttatlas-assign-slots --ttatlas-verify-contract=stage=6 -o %t.row_min_16
// RUN: FileCheck %s --input-file=%t.row_min_16 --check-prefix=row_min_16
// RUN: triton-atlas-opt %S/../inputs/stage3/row_reduce.mlir --ttatlas-legalize-dtypes --canonicalize --cse --ttatlas-legalize-shapes --ttatlas-assign-slots --ttatlas-verify-contract=stage=6 -o %t.row_reduce
// RUN: FileCheck %s --input-file=%t.row_reduce --check-prefix=row_reduce
// RUN: triton-atlas-opt %S/../inputs/stage3/row_sum_scaled_64x16.mlir --ttatlas-legalize-dtypes --canonicalize --cse --ttatlas-legalize-shapes --ttatlas-assign-slots --ttatlas-verify-contract=stage=6 -o %t.row_sum_scaled_64x16
// RUN: FileCheck %s --input-file=%t.row_sum_scaled_64x16 --check-prefix=row_sum_scaled_64x16
// RUN: triton-atlas-opt %S/../inputs/stage3/rowsum_plus_bias.mlir --ttatlas-legalize-dtypes --canonicalize --cse --ttatlas-legalize-shapes --ttatlas-assign-slots --ttatlas-verify-contract=stage=6 -o %t.rowsum_plus_bias
// RUN: FileCheck %s --input-file=%t.rowsum_plus_bias --check-prefix=rowsum_plus_bias
// RUN: triton-atlas-opt %S/../inputs/stage3/softmax_16.mlir --ttatlas-legalize-dtypes --canonicalize --cse --ttatlas-legalize-shapes --ttatlas-assign-slots --ttatlas-verify-contract=stage=6 -o %t.softmax_16
// RUN: FileCheck %s --input-file=%t.softmax_16 --check-prefix=softmax_16
// RUN: triton-atlas-opt %S/../inputs/stage3/softmax_masked_20.mlir --ttatlas-legalize-dtypes --canonicalize --cse --ttatlas-legalize-shapes --ttatlas-assign-slots --ttatlas-verify-contract=stage=6 -o %t.softmax_masked_20
// RUN: FileCheck %s --input-file=%t.softmax_masked_20 --check-prefix=softmax_masked_20
// RUN: triton-atlas-opt %S/../inputs/stage3/strided_cols.mlir --ttatlas-legalize-dtypes --canonicalize --cse --ttatlas-legalize-shapes --ttatlas-assign-slots --ttatlas-verify-contract=stage=6 -o %t.strided_cols
// RUN: FileCheck %s --input-file=%t.strided_cols --check-prefix=strided_cols
// RUN: triton-atlas-opt %S/../inputs/stage3/tail_mask_k_loop.mlir --ttatlas-legalize-dtypes --canonicalize --cse --ttatlas-legalize-shapes --ttatlas-assign-slots --ttatlas-verify-contract=stage=6 -o %t.tail_mask_k_loop
// RUN: FileCheck %s --input-file=%t.tail_mask_k_loop --check-prefix=tail_mask_k_loop
// RUN: triton-atlas-opt %S/../inputs/stage3/unrolled_by_triton.mlir --ttatlas-legalize-dtypes --canonicalize --cse --ttatlas-legalize-shapes --ttatlas-assign-slots --ttatlas-verify-contract=stage=6 -o %t.unrolled_by_triton
// RUN: FileCheck %s --input-file=%t.unrolled_by_triton --check-prefix=unrolled_by_triton

// RUN: triton-atlas-opt %S/../inputs/stage3/copy_rows_50.mlir --ttatlas-legalize-dtypes --canonicalize --cse --ttatlas-legalize-shapes --ttatlas-assign-slots --ttatlas-verify-contract=stage=6 -o %t.copy_rows_50
// RUN: FileCheck %s --input-file=%t.copy_rows_50 --check-prefix=copy_rows_50
// RUN: triton-atlas-opt %S/../inputs/stage3/bias_both_layouts.mlir --ttatlas-legalize-dtypes --canonicalize --cse --ttatlas-legalize-shapes --ttatlas-assign-slots --ttatlas-verify-contract=stage=6 -o %t.bias_both_layouts
// RUN: FileCheck %s --input-file=%t.bias_both_layouts --check-prefix=bias_both_layouts
// Pass 6 on the probe kernels after passes 5 and 4, with the stage-6 contract checked on each result. Each count is the
// number of slots, then the number of ops carrying them (a pack, an unpack, a constant, or a copy's load or store).
// Slots: one per tile of each pack and unpack (identical input tiles share a slot index, but each carries its
// attribute), one per tile of each clamp constant, one per tt.load / tt.store of a copy. Details of the slot fields
// are in slots.mlir; what pass 6 rejects is in reject.mlir.

// add_16 has 2 slot(s) on 2 op(s)
// add_16-COUNT-2: ttatlas.slot =
// add_16-NOT: ttatlas.slot =

// add_50x40_of_64 has 8 slot(s) on 2 op(s)
// add_50x40_of_64-COUNT-2: ttatlas.slot =
// add_50x40_of_64-NOT: ttatlas.slot =

// add_64x16 has 4 slot(s) on 2 op(s)
// add_64x16-COUNT-2: ttatlas.slot =
// add_64x16-NOT: ttatlas.slot =

// add_rows_50 has 4 slot(s) on 2 op(s)
// add_rows_50-COUNT-2: ttatlas.slot =
// add_rows_50-NOT: ttatlas.slot =

// bias_as_column has 3 slot(s) on 3 op(s)
// bias_as_column-COUNT-3: ttatlas.slot =
// bias_as_column-NOT: ttatlas.slot =

// bias_seeded_k_loop has 10 slot(s) on 10 op(s)
// bias_seeded_k_loop-COUNT-10: ttatlas.slot =
// bias_seeded_k_loop-NOT: ttatlas.slot =

// copy_16 has 2 slot(s) on 2 op(s)
// copy_16-COUNT-2: ttatlas.slot =
// copy_16-NOT: ttatlas.slot =

// dot_plus_dot has 5 slot(s) on 5 op(s)
// dot_plus_dot-COUNT-5: ttatlas.slot =
// dot_plus_dot-NOT: ttatlas.slot =

// masked_rows_50_matmul has 5 slot(s) on 3 op(s)
// masked_rows_50_matmul-COUNT-3: ttatlas.slot =
// masked_rows_50_matmul-NOT: ttatlas.slot =

// matmul_64 has 12 slot(s) on 3 op(s)
// matmul_64-COUNT-3: ttatlas.slot =
// matmul_64-NOT: ttatlas.slot =

// matmul_relu has 3 slot(s) on 3 op(s)
// matmul_relu-COUNT-3: ttatlas.slot =
// matmul_relu-NOT: ttatlas.slot =

// matmul_relu_64x32 has 5 slot(s) on 3 op(s)
// matmul_relu_64x32-COUNT-3: ttatlas.slot =
// matmul_relu_64x32-NOT: ttatlas.slot =

// mlp has 4 slot(s) on 4 op(s)
// mlp-COUNT-4: ttatlas.slot =
// mlp-NOT: ttatlas.slot =

// mm32 has 9 slot(s) on 9 op(s)
// mm32-COUNT-9: ttatlas.slot =
// mm32-NOT: ttatlas.slot =

// row_max_16 has 3 slot(s) on 3 op(s)
// row_max_16-COUNT-3: ttatlas.slot =
// row_max_16-NOT: ttatlas.slot =

// row_max_64 has 6 slot(s) on 2 op(s)
// row_max_64-COUNT-2: ttatlas.slot =
// row_max_64-NOT: ttatlas.slot =

// row_min_16 has 3 slot(s) on 3 op(s)
// row_min_16-COUNT-3: ttatlas.slot =
// row_min_16-NOT: ttatlas.slot =

// row_reduce has 2 slot(s) on 2 op(s)
// row_reduce-COUNT-2: ttatlas.slot =
// row_reduce-NOT: ttatlas.slot =

// row_sum_scaled_64x16 has 8 slot(s) on 4 op(s)
// row_sum_scaled_64x16-COUNT-4: ttatlas.slot =
// row_sum_scaled_64x16-NOT: ttatlas.slot =

// rowsum_plus_bias has 3 slot(s) on 3 op(s)
// rowsum_plus_bias-COUNT-3: ttatlas.slot =
// rowsum_plus_bias-NOT: ttatlas.slot =

// softmax_16 has 5 slot(s) on 5 op(s)
// softmax_16-COUNT-5: ttatlas.slot =
// softmax_16-NOT: ttatlas.slot =

// softmax_masked_20 has 2 slot(s) on 2 op(s)
// softmax_masked_20-COUNT-2: ttatlas.slot =
// softmax_masked_20-NOT: ttatlas.slot =

// strided_cols has 2 slot(s) on 2 op(s)
// strided_cols-COUNT-2: ttatlas.slot =
// strided_cols-NOT: ttatlas.slot =

// tail_mask_k_loop has 3 slot(s) on 3 op(s)
// tail_mask_k_loop-COUNT-3: ttatlas.slot =
// tail_mask_k_loop-NOT: ttatlas.slot =

// unrolled_by_triton has 9 slot(s) on 9 op(s)
// unrolled_by_triton-COUNT-9: ttatlas.slot =
// unrolled_by_triton-NOT: ttatlas.slot =

// copy_rows_50 has 2 slot(s) on 2 op(s)
// copy_rows_50-COUNT-2: ttatlas.slot =
// copy_rows_50-NOT: ttatlas.slot =

// bias_both_layouts has 4 slot(s) on 4 op(s)
// bias_both_layouts-COUNT-4: ttatlas.slot =
// bias_both_layouts-NOT: ttatlas.slot =
