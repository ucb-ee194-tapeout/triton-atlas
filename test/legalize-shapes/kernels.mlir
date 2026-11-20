// RUN: triton-atlas-opt %S/../inputs/stage3/add_16.mlir --ttatlas-legalize-dtypes --canonicalize --cse --ttatlas-legalize-shapes --ttatlas-verify-contract=stage=4 -o %t.add_16
// RUN: FileCheck %s --input-file=%t.add_16 --check-prefix=add_16_pack
// RUN: FileCheck %s --input-file=%t.add_16 --check-prefix=add_16_unpack
// RUN: FileCheck %s --input-file=%t.add_16 --check-prefix=add_16_red
// RUN: triton-atlas-opt %S/../inputs/stage3/add_50x40_of_64.mlir --ttatlas-legalize-dtypes --canonicalize --cse --ttatlas-legalize-shapes --ttatlas-verify-contract=stage=4 -o %t.add_50x40_of_64
// RUN: FileCheck %s --input-file=%t.add_50x40_of_64 --check-prefix=add_50x40_of_64_pack
// RUN: FileCheck %s --input-file=%t.add_50x40_of_64 --check-prefix=add_50x40_of_64_unpack
// RUN: FileCheck %s --input-file=%t.add_50x40_of_64 --check-prefix=add_50x40_of_64_red
// RUN: triton-atlas-opt %S/../inputs/stage3/add_64x16.mlir --ttatlas-legalize-dtypes --canonicalize --cse --ttatlas-legalize-shapes --ttatlas-verify-contract=stage=4 -o %t.add_64x16
// RUN: FileCheck %s --input-file=%t.add_64x16 --check-prefix=add_64x16_pack
// RUN: FileCheck %s --input-file=%t.add_64x16 --check-prefix=add_64x16_unpack
// RUN: FileCheck %s --input-file=%t.add_64x16 --check-prefix=add_64x16_red
// RUN: triton-atlas-opt %S/../inputs/stage3/add_rows_50.mlir --ttatlas-legalize-dtypes --canonicalize --cse --ttatlas-legalize-shapes --ttatlas-verify-contract=stage=4 -o %t.add_rows_50
// RUN: FileCheck %s --input-file=%t.add_rows_50 --check-prefix=add_rows_50_pack
// RUN: FileCheck %s --input-file=%t.add_rows_50 --check-prefix=add_rows_50_unpack
// RUN: FileCheck %s --input-file=%t.add_rows_50 --check-prefix=add_rows_50_red
// RUN: triton-atlas-opt %S/../inputs/stage3/bias_as_column.mlir --ttatlas-legalize-dtypes --canonicalize --cse --ttatlas-legalize-shapes --ttatlas-verify-contract=stage=4 -o %t.bias_as_column
// RUN: FileCheck %s --input-file=%t.bias_as_column --check-prefix=bias_as_column_pack
// RUN: FileCheck %s --input-file=%t.bias_as_column --check-prefix=bias_as_column_unpack
// RUN: FileCheck %s --input-file=%t.bias_as_column --check-prefix=bias_as_column_red
// RUN: triton-atlas-opt %S/../inputs/stage3/bias_seeded_k_loop.mlir --ttatlas-legalize-dtypes --canonicalize --cse --ttatlas-legalize-shapes --ttatlas-verify-contract=stage=4 -o %t.bias_seeded_k_loop
// RUN: FileCheck %s --input-file=%t.bias_seeded_k_loop --check-prefix=bias_seeded_k_loop_pack
// RUN: FileCheck %s --input-file=%t.bias_seeded_k_loop --check-prefix=bias_seeded_k_loop_unpack
// RUN: FileCheck %s --input-file=%t.bias_seeded_k_loop --check-prefix=bias_seeded_k_loop_red
// RUN: triton-atlas-opt %S/../inputs/stage3/colbroadcast_of_rowsum.mlir --ttatlas-legalize-dtypes --canonicalize --cse --ttatlas-legalize-shapes --ttatlas-verify-contract=stage=4 -o %t.colbroadcast_of_rowsum
// RUN: FileCheck %s --input-file=%t.colbroadcast_of_rowsum --check-prefix=colbroadcast_of_rowsum_pack
// RUN: FileCheck %s --input-file=%t.colbroadcast_of_rowsum --check-prefix=colbroadcast_of_rowsum_unpack
// RUN: FileCheck %s --input-file=%t.colbroadcast_of_rowsum --check-prefix=colbroadcast_of_rowsum_red
// RUN: triton-atlas-opt %S/../inputs/stage3/copy_16.mlir --ttatlas-legalize-dtypes --canonicalize --cse --ttatlas-legalize-shapes --ttatlas-verify-contract=stage=4 -o %t.copy_16
// RUN: FileCheck %s --input-file=%t.copy_16 --check-prefix=copy_16_pack
// RUN: FileCheck %s --input-file=%t.copy_16 --check-prefix=copy_16_unpack
// RUN: FileCheck %s --input-file=%t.copy_16 --check-prefix=copy_16_red
// RUN: triton-atlas-opt %S/../inputs/stage3/copy_rows_50.mlir --ttatlas-legalize-dtypes --canonicalize --cse --ttatlas-legalize-shapes --ttatlas-verify-contract=stage=4 -o %t.copy_rows_50
// RUN: FileCheck %s --input-file=%t.copy_rows_50 --check-prefix=copy_rows_50_pack
// RUN: FileCheck %s --input-file=%t.copy_rows_50 --check-prefix=copy_rows_50_unpack
// RUN: FileCheck %s --input-file=%t.copy_rows_50 --check-prefix=copy_rows_50_red
// RUN: triton-atlas-opt %S/../inputs/stage3/dot_plus_dot.mlir --ttatlas-legalize-dtypes --canonicalize --cse --ttatlas-legalize-shapes --ttatlas-verify-contract=stage=4 -o %t.dot_plus_dot
// RUN: FileCheck %s --input-file=%t.dot_plus_dot --check-prefix=dot_plus_dot_pack
// RUN: FileCheck %s --input-file=%t.dot_plus_dot --check-prefix=dot_plus_dot_unpack
// RUN: FileCheck %s --input-file=%t.dot_plus_dot --check-prefix=dot_plus_dot_red
// RUN: triton-atlas-opt %S/../inputs/stage3/masked_rows_50_matmul.mlir --ttatlas-legalize-dtypes --canonicalize --cse --ttatlas-legalize-shapes --ttatlas-verify-contract=stage=4 -o %t.masked_rows_50_matmul
// RUN: FileCheck %s --input-file=%t.masked_rows_50_matmul --check-prefix=masked_rows_50_matmul_pack
// RUN: FileCheck %s --input-file=%t.masked_rows_50_matmul --check-prefix=masked_rows_50_matmul_unpack
// RUN: FileCheck %s --input-file=%t.masked_rows_50_matmul --check-prefix=masked_rows_50_matmul_red
// RUN: triton-atlas-opt %S/../inputs/stage3/matmul_64.mlir --ttatlas-legalize-dtypes --canonicalize --cse --ttatlas-legalize-shapes --ttatlas-verify-contract=stage=4 -o %t.matmul_64
// RUN: FileCheck %s --input-file=%t.matmul_64 --check-prefix=matmul_64_pack
// RUN: FileCheck %s --input-file=%t.matmul_64 --check-prefix=matmul_64_unpack
// RUN: FileCheck %s --input-file=%t.matmul_64 --check-prefix=matmul_64_red
// RUN: triton-atlas-opt %S/../inputs/stage3/matmul_relu.mlir --ttatlas-legalize-dtypes --canonicalize --cse --ttatlas-legalize-shapes --ttatlas-verify-contract=stage=4 -o %t.matmul_relu
// RUN: FileCheck %s --input-file=%t.matmul_relu --check-prefix=matmul_relu_pack
// RUN: FileCheck %s --input-file=%t.matmul_relu --check-prefix=matmul_relu_unpack
// RUN: FileCheck %s --input-file=%t.matmul_relu --check-prefix=matmul_relu_red
// RUN: triton-atlas-opt %S/../inputs/stage3/matmul_relu_64x32.mlir --ttatlas-legalize-dtypes --canonicalize --cse --ttatlas-legalize-shapes --ttatlas-verify-contract=stage=4 -o %t.matmul_relu_64x32
// RUN: FileCheck %s --input-file=%t.matmul_relu_64x32 --check-prefix=matmul_relu_64x32_pack
// RUN: FileCheck %s --input-file=%t.matmul_relu_64x32 --check-prefix=matmul_relu_64x32_unpack
// RUN: FileCheck %s --input-file=%t.matmul_relu_64x32 --check-prefix=matmul_relu_64x32_red
// RUN: triton-atlas-opt %S/../inputs/stage3/mlp.mlir --ttatlas-legalize-dtypes --canonicalize --cse --ttatlas-legalize-shapes --ttatlas-verify-contract=stage=4 -o %t.mlp
// RUN: FileCheck %s --input-file=%t.mlp --check-prefix=mlp_pack
// RUN: FileCheck %s --input-file=%t.mlp --check-prefix=mlp_unpack
// RUN: FileCheck %s --input-file=%t.mlp --check-prefix=mlp_red
// RUN: triton-atlas-opt %S/../inputs/stage3/mm32.mlir --ttatlas-legalize-dtypes --canonicalize --cse --ttatlas-legalize-shapes --ttatlas-verify-contract=stage=4 -o %t.mm32
// RUN: FileCheck %s --input-file=%t.mm32 --check-prefix=mm32_pack
// RUN: FileCheck %s --input-file=%t.mm32 --check-prefix=mm32_unpack
// RUN: FileCheck %s --input-file=%t.mm32 --check-prefix=mm32_red
// RUN: triton-atlas-opt %S/../inputs/stage3/row_max_64.mlir --ttatlas-legalize-dtypes --canonicalize --cse --ttatlas-legalize-shapes --ttatlas-verify-contract=stage=4 -o %t.row_max_64
// RUN: FileCheck %s --input-file=%t.row_max_64 --check-prefix=row_max_64_pack
// RUN: FileCheck %s --input-file=%t.row_max_64 --check-prefix=row_max_64_unpack
// RUN: FileCheck %s --input-file=%t.row_max_64 --check-prefix=row_max_64_red
// RUN: triton-atlas-opt %S/../inputs/stage3/row_reduce.mlir --ttatlas-legalize-dtypes --canonicalize --cse --ttatlas-legalize-shapes --ttatlas-verify-contract=stage=4 -o %t.row_reduce
// RUN: FileCheck %s --input-file=%t.row_reduce --check-prefix=row_reduce_pack
// RUN: FileCheck %s --input-file=%t.row_reduce --check-prefix=row_reduce_unpack
// RUN: FileCheck %s --input-file=%t.row_reduce --check-prefix=row_reduce_red
// RUN: triton-atlas-opt %S/../inputs/stage3/row_sum_scaled_64x16.mlir --ttatlas-legalize-dtypes --canonicalize --cse --ttatlas-legalize-shapes --ttatlas-verify-contract=stage=4 -o %t.row_sum_scaled_64x16
// RUN: FileCheck %s --input-file=%t.row_sum_scaled_64x16 --check-prefix=row_sum_scaled_64x16_pack
// RUN: FileCheck %s --input-file=%t.row_sum_scaled_64x16 --check-prefix=row_sum_scaled_64x16_unpack
// RUN: FileCheck %s --input-file=%t.row_sum_scaled_64x16 --check-prefix=row_sum_scaled_64x16_red
// RUN: triton-atlas-opt %S/../inputs/stage3/rowsum_plus_bias.mlir --ttatlas-legalize-dtypes --canonicalize --cse --ttatlas-legalize-shapes --ttatlas-verify-contract=stage=4 -o %t.rowsum_plus_bias
// RUN: FileCheck %s --input-file=%t.rowsum_plus_bias --check-prefix=rowsum_plus_bias_pack
// RUN: FileCheck %s --input-file=%t.rowsum_plus_bias --check-prefix=rowsum_plus_bias_unpack
// RUN: FileCheck %s --input-file=%t.rowsum_plus_bias --check-prefix=rowsum_plus_bias_red
// RUN: triton-atlas-opt %S/../inputs/stage3/softmax_16.mlir --ttatlas-legalize-dtypes --canonicalize --cse --ttatlas-legalize-shapes --ttatlas-verify-contract=stage=4 -o %t.softmax_16
// RUN: FileCheck %s --input-file=%t.softmax_16 --check-prefix=softmax_16_pack
// RUN: FileCheck %s --input-file=%t.softmax_16 --check-prefix=softmax_16_unpack
// RUN: FileCheck %s --input-file=%t.softmax_16 --check-prefix=softmax_16_red
// RUN: triton-atlas-opt %S/../inputs/stage3/softmax_masked_20.mlir --ttatlas-legalize-dtypes --canonicalize --cse --ttatlas-legalize-shapes --ttatlas-verify-contract=stage=4 -o %t.softmax_masked_20
// RUN: FileCheck %s --input-file=%t.softmax_masked_20 --check-prefix=softmax_masked_20_pack
// RUN: FileCheck %s --input-file=%t.softmax_masked_20 --check-prefix=softmax_masked_20_unpack
// RUN: FileCheck %s --input-file=%t.softmax_masked_20 --check-prefix=softmax_masked_20_red
// RUN: triton-atlas-opt %S/../inputs/stage3/strided_cols.mlir --ttatlas-legalize-dtypes --canonicalize --cse --ttatlas-legalize-shapes --ttatlas-verify-contract=stage=4 -o %t.strided_cols
// RUN: FileCheck %s --input-file=%t.strided_cols --check-prefix=strided_cols_pack
// RUN: FileCheck %s --input-file=%t.strided_cols --check-prefix=strided_cols_unpack
// RUN: FileCheck %s --input-file=%t.strided_cols --check-prefix=strided_cols_red
// RUN: triton-atlas-opt %S/../inputs/stage3/tail_mask_k_loop.mlir --ttatlas-legalize-dtypes --canonicalize --cse --ttatlas-legalize-shapes --ttatlas-verify-contract=stage=4 -o %t.tail_mask_k_loop
// RUN: FileCheck %s --input-file=%t.tail_mask_k_loop --check-prefix=tail_mask_k_loop_pack
// RUN: FileCheck %s --input-file=%t.tail_mask_k_loop --check-prefix=tail_mask_k_loop_unpack
// RUN: FileCheck %s --input-file=%t.tail_mask_k_loop --check-prefix=tail_mask_k_loop_red
// RUN: triton-atlas-opt %S/../inputs/stage3/unrolled_by_triton.mlir --ttatlas-legalize-dtypes --canonicalize --cse --ttatlas-legalize-shapes --ttatlas-verify-contract=stage=4 -o %t.unrolled_by_triton
// RUN: FileCheck %s --input-file=%t.unrolled_by_triton --check-prefix=unrolled_by_triton_pack
// RUN: FileCheck %s --input-file=%t.unrolled_by_triton --check-prefix=unrolled_by_triton_unpack
// RUN: FileCheck %s --input-file=%t.unrolled_by_triton --check-prefix=unrolled_by_triton_red

// Pass 4 (steps 4a-4d) on the 25 probe kernels, after pass 5, with the stage-4 contract checked on each result. The
// counts say what the design promises: one pack per load and one unpack per store, nothing left between compute ops
// (also when the block is padded), and a contraction or row reduction over several tiles as a chain of tile ops. A
// kernel with no linalg op (a plain copy) keeps its tt.load and tt.store.

// add_16: 1 pack(s), 1 unpack(s), 0 reducing generic(s)
// add_16_pack-COUNT-1: linalg.pack
// add_16_pack-NOT: linalg.pack
// add_16_unpack-COUNT-1: linalg.unpack
// add_16_unpack-NOT: linalg.unpack
// add_16_red-NOT: iterator_types = [{{.*}}"reduction"

// add_50x40_of_64: 1 pack(s), 1 unpack(s), 0 reducing generic(s)
// add_50x40_of_64_pack-COUNT-1: linalg.pack
// add_50x40_of_64_pack-NOT: linalg.pack
// add_50x40_of_64_unpack-COUNT-1: linalg.unpack
// add_50x40_of_64_unpack-NOT: linalg.unpack
// add_50x40_of_64_red-NOT: iterator_types = [{{.*}}"reduction"

// add_64x16: 1 pack(s), 1 unpack(s), 0 reducing generic(s)
// add_64x16_pack-COUNT-1: linalg.pack
// add_64x16_pack-NOT: linalg.pack
// add_64x16_unpack-COUNT-1: linalg.unpack
// add_64x16_unpack-NOT: linalg.unpack
// add_64x16_red-NOT: iterator_types = [{{.*}}"reduction"

// add_rows_50: 1 pack(s), 1 unpack(s), 0 reducing generic(s)
// add_rows_50_pack-COUNT-1: linalg.pack
// add_rows_50_pack-NOT: linalg.pack
// add_rows_50_unpack-COUNT-1: linalg.unpack
// add_rows_50_unpack-NOT: linalg.unpack
// add_rows_50_red-NOT: iterator_types = [{{.*}}"reduction"

// bias_as_column: 2 pack(s), 1 unpack(s), 0 reducing generic(s)
// bias_as_column_pack-COUNT-2: linalg.pack
// bias_as_column_pack-NOT: linalg.pack
// bias_as_column_unpack-COUNT-1: linalg.unpack
// bias_as_column_unpack-NOT: linalg.unpack
// bias_as_column_red-NOT: iterator_types = [{{.*}}"reduction"

// bias_seeded_k_loop: 9 pack(s), 1 unpack(s), 4 reducing generic(s)
// bias_seeded_k_loop_pack-COUNT-9: linalg.pack
// bias_seeded_k_loop_pack-NOT: linalg.pack
// bias_seeded_k_loop_unpack-COUNT-1: linalg.unpack
// bias_seeded_k_loop_unpack-NOT: linalg.unpack
// bias_seeded_k_loop_red-COUNT-4: iterator_types = [{{.*}}"reduction"
// bias_seeded_k_loop_red-NOT: iterator_types = [{{.*}}"reduction"

// colbroadcast_of_rowsum: 1 pack(s), 1 unpack(s), 1 reducing generic(s)
// colbroadcast_of_rowsum_pack-COUNT-1: linalg.pack
// colbroadcast_of_rowsum_pack-NOT: linalg.pack
// colbroadcast_of_rowsum_unpack-COUNT-1: linalg.unpack
// colbroadcast_of_rowsum_unpack-NOT: linalg.unpack
// colbroadcast_of_rowsum_red-COUNT-1: iterator_types = [{{.*}}"reduction"
// colbroadcast_of_rowsum_red-NOT: iterator_types = [{{.*}}"reduction"

// copy_16: 0 pack(s), 0 unpack(s), 0 reducing generic(s)
// copy_16_pack-NOT: linalg.pack
// copy_16_unpack-NOT: linalg.unpack
// copy_16_red-NOT: iterator_types = [{{.*}}"reduction"

// copy_rows_50: 0 pack(s), 0 unpack(s), 0 reducing generic(s)
// copy_rows_50_pack-NOT: linalg.pack
// copy_rows_50_unpack-NOT: linalg.unpack
// copy_rows_50_red-NOT: iterator_types = [{{.*}}"reduction"

// dot_plus_dot: 4 pack(s), 1 unpack(s), 2 reducing generic(s)
// dot_plus_dot_pack-COUNT-4: linalg.pack
// dot_plus_dot_pack-NOT: linalg.pack
// dot_plus_dot_unpack-COUNT-1: linalg.unpack
// dot_plus_dot_unpack-NOT: linalg.unpack
// dot_plus_dot_red-COUNT-2: iterator_types = [{{.*}}"reduction"
// dot_plus_dot_red-NOT: iterator_types = [{{.*}}"reduction"

// masked_rows_50_matmul: 2 pack(s), 1 unpack(s), 2 reducing generic(s)
// masked_rows_50_matmul_pack-COUNT-2: linalg.pack
// masked_rows_50_matmul_pack-NOT: linalg.pack
// masked_rows_50_matmul_unpack-COUNT-1: linalg.unpack
// masked_rows_50_matmul_unpack-NOT: linalg.unpack
// masked_rows_50_matmul_red-COUNT-2: iterator_types = [{{.*}}"reduction"
// masked_rows_50_matmul_red-NOT: iterator_types = [{{.*}}"reduction"

// matmul_64: 2 pack(s), 1 unpack(s), 8 reducing generic(s)
// matmul_64_pack-COUNT-2: linalg.pack
// matmul_64_pack-NOT: linalg.pack
// matmul_64_unpack-COUNT-1: linalg.unpack
// matmul_64_unpack-NOT: linalg.unpack
// matmul_64_red-COUNT-8: iterator_types = [{{.*}}"reduction"
// matmul_64_red-NOT: iterator_types = [{{.*}}"reduction"

// matmul_relu: 2 pack(s), 1 unpack(s), 1 reducing generic(s)
// matmul_relu_pack-COUNT-2: linalg.pack
// matmul_relu_pack-NOT: linalg.pack
// matmul_relu_unpack-COUNT-1: linalg.unpack
// matmul_relu_unpack-NOT: linalg.unpack
// matmul_relu_red-COUNT-1: iterator_types = [{{.*}}"reduction"
// matmul_relu_red-NOT: iterator_types = [{{.*}}"reduction"

// matmul_relu_64x32: 2 pack(s), 1 unpack(s), 2 reducing generic(s)
// matmul_relu_64x32_pack-COUNT-2: linalg.pack
// matmul_relu_64x32_pack-NOT: linalg.pack
// matmul_relu_64x32_unpack-COUNT-1: linalg.unpack
// matmul_relu_64x32_unpack-NOT: linalg.unpack
// matmul_relu_64x32_red-COUNT-2: iterator_types = [{{.*}}"reduction"
// matmul_relu_64x32_red-NOT: iterator_types = [{{.*}}"reduction"

// mlp: 3 pack(s), 1 unpack(s), 1 reducing generic(s)
// mlp_pack-COUNT-3: linalg.pack
// mlp_pack-NOT: linalg.pack
// mlp_unpack-COUNT-1: linalg.unpack
// mlp_unpack-NOT: linalg.unpack
// mlp_red-COUNT-1: iterator_types = [{{.*}}"reduction"
// mlp_red-NOT: iterator_types = [{{.*}}"reduction"

// mm32: 8 pack(s), 1 unpack(s), 4 reducing generic(s)
// mm32_pack-COUNT-8: linalg.pack
// mm32_pack-NOT: linalg.pack
// mm32_unpack-COUNT-1: linalg.unpack
// mm32_unpack-NOT: linalg.unpack
// mm32_red-COUNT-4: iterator_types = [{{.*}}"reduction"
// mm32_red-NOT: iterator_types = [{{.*}}"reduction"

// row_max_64: 1 pack(s), 1 unpack(s), 4 reducing generic(s)
// row_max_64_pack-COUNT-1: linalg.pack
// row_max_64_pack-NOT: linalg.pack
// row_max_64_unpack-COUNT-1: linalg.unpack
// row_max_64_unpack-NOT: linalg.unpack
// row_max_64_red-COUNT-4: iterator_types = [{{.*}}"reduction"
// row_max_64_red-NOT: iterator_types = [{{.*}}"reduction"

// row_reduce: 1 pack(s), 1 unpack(s), 2 reducing generic(s)
// row_reduce_pack-COUNT-1: linalg.pack
// row_reduce_pack-NOT: linalg.pack
// row_reduce_unpack-COUNT-1: linalg.unpack
// row_reduce_unpack-NOT: linalg.unpack
// row_reduce_red-COUNT-2: iterator_types = [{{.*}}"reduction"
// row_reduce_red-NOT: iterator_types = [{{.*}}"reduction"

// row_sum_scaled_64x16: 1 pack(s), 1 unpack(s), 2 reducing generic(s)
// row_sum_scaled_64x16_pack-COUNT-1: linalg.pack
// row_sum_scaled_64x16_pack-NOT: linalg.pack
// row_sum_scaled_64x16_unpack-COUNT-1: linalg.unpack
// row_sum_scaled_64x16_unpack-NOT: linalg.unpack
// row_sum_scaled_64x16_red-COUNT-2: iterator_types = [{{.*}}"reduction"
// row_sum_scaled_64x16_red-NOT: iterator_types = [{{.*}}"reduction"

// rowsum_plus_bias: 2 pack(s), 1 unpack(s), 1 reducing generic(s)
// rowsum_plus_bias_pack-COUNT-2: linalg.pack
// rowsum_plus_bias_pack-NOT: linalg.pack
// rowsum_plus_bias_unpack-COUNT-1: linalg.unpack
// rowsum_plus_bias_unpack-NOT: linalg.unpack
// rowsum_plus_bias_red-COUNT-1: iterator_types = [{{.*}}"reduction"
// rowsum_plus_bias_red-NOT: iterator_types = [{{.*}}"reduction"

// softmax_16: 1 pack(s), 1 unpack(s), 2 reducing generic(s)
// softmax_16_pack-COUNT-1: linalg.pack
// softmax_16_pack-NOT: linalg.pack
// softmax_16_unpack-COUNT-1: linalg.unpack
// softmax_16_unpack-NOT: linalg.unpack
// softmax_16_red-COUNT-2: iterator_types = [{{.*}}"reduction"
// softmax_16_red-NOT: iterator_types = [{{.*}}"reduction"

// softmax_masked_20: 1 pack(s), 1 unpack(s), 2 reducing generic(s)
// softmax_masked_20_pack-COUNT-1: linalg.pack
// softmax_masked_20_pack-NOT: linalg.pack
// softmax_masked_20_unpack-COUNT-1: linalg.unpack
// softmax_masked_20_unpack-NOT: linalg.unpack
// softmax_masked_20_red-COUNT-2: iterator_types = [{{.*}}"reduction"
// softmax_masked_20_red-NOT: iterator_types = [{{.*}}"reduction"

// strided_cols: 0 pack(s), 0 unpack(s), 0 reducing generic(s)
// strided_cols_pack-NOT: linalg.pack
// strided_cols_unpack-NOT: linalg.unpack
// strided_cols_red-NOT: iterator_types = [{{.*}}"reduction"

// tail_mask_k_loop: 2 pack(s), 1 unpack(s), 0 reducing generic(s)
// tail_mask_k_loop_pack-COUNT-2: linalg.pack
// tail_mask_k_loop_pack-NOT: linalg.pack
// tail_mask_k_loop_unpack-COUNT-1: linalg.unpack
// tail_mask_k_loop_unpack-NOT: linalg.unpack
// tail_mask_k_loop_red-NOT: iterator_types = [{{.*}}"reduction"

// unrolled_by_triton: 8 pack(s), 1 unpack(s), 4 reducing generic(s)
// unrolled_by_triton_pack-COUNT-8: linalg.pack
// unrolled_by_triton_pack-NOT: linalg.pack
// unrolled_by_triton_unpack-COUNT-1: linalg.unpack
// unrolled_by_triton_unpack-NOT: linalg.unpack
// unrolled_by_triton_red-COUNT-4: iterator_types = [{{.*}}"reduction"
// unrolled_by_triton_red-NOT: iterator_types = [{{.*}}"reduction"
