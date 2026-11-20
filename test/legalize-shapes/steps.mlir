// RUN: triton-atlas-opt %S/../inputs/stage3/softmax_16.mlir --ttatlas-legalize-dtypes --canonicalize --cse --ttatlas-pack-blocks --canonicalize --cse -o %t.pack
// RUN: FileCheck %s --input-file=%t.pack --check-prefix=PACK
// RUN: triton-atlas-opt %S/../inputs/stage3/row_max_16.mlir --ttatlas-legalize-dtypes --canonicalize --cse --ttatlas-pack-blocks -o %t.max
// RUN: FileCheck %s --input-file=%t.max --check-prefix=MAX
// RUN: triton-atlas-opt %S/../inputs/stage3/row_min_16.mlir --ttatlas-legalize-dtypes --canonicalize --cse --ttatlas-pack-blocks -o %t.min
// RUN: FileCheck %s --input-file=%t.min --check-prefix=MIN
// RUN: triton-atlas-opt %S/../inputs/stage3/softmax_masked_20.mlir --ttatlas-legalize-dtypes --canonicalize --cse --ttatlas-pack-blocks -o %t.whole
// RUN: FileCheck %s --input-file=%t.whole --check-prefix=WHOLE
// RUN: triton-atlas-opt %S/../inputs/stage3/mlp.mlir --ttatlas-legalize-dtypes --canonicalize --cse --ttatlas-pack-blocks --canonicalize -o %t.mlp
// RUN: FileCheck %s --input-file=%t.mlp --check-prefix=MLP
// RUN: triton-atlas-opt %S/../inputs/stage3/softmax_16.mlir --ttatlas-legalize-dtypes --canonicalize --cse --ttatlas-pack-blocks --canonicalize --cse --ttatlas-fold-pack-pairs --canonicalize --cse -o %t.fold
// RUN: FileCheck %s --input-file=%t.fold --check-prefix=FOLD
// RUN: triton-atlas-opt %S/../inputs/stage3/matmul_64.mlir --ttatlas-legalize-dtypes --canonicalize --cse --ttatlas-pack-blocks --canonicalize --cse --ttatlas-fold-pack-pairs --canonicalize --cse --ttatlas-tile-grid --canonicalize --cse -o %t.tile
// RUN: FileCheck %s --input-file=%t.tile --check-prefix=TILE
// RUN: triton-atlas-opt %S/../inputs/stage3/mm32.mlir --ttatlas-legalize-dtypes --canonicalize --cse --ttatlas-legalize-shapes -o %t.units
// RUN: FileCheck %s --input-file=%t.units --check-prefix=UNITS

// Step 4a (ttatlas-pack-blocks) on a padded 16x16 softmax: every generic is packed in 32x32 blocks, with no
// outer_dims_perm, the pad of the packs is zero, and each padded row reduction is read through a clamp generic.
// PACK-NOT: outer_dims_perm
// A row max reads x' = min(x, high): in-block lanes 0x7FFF (bytes FF 7F), pad lanes -inf (0xFF80, bytes 80 FF).
// PACK-DAG: arith.constant dense<"0xFF7FFF7F
// A row sum reads x' = max(min(x, high), low): the same generic holds both; pad lanes are 0.
// PACK-DAG: arith.constant dense<"0xFFFFFFFF
// PACK: linalg.pack {{.*}}padding_value
// PACK: arith.minnumf
// PACK: arith.minnumf
// PACK-NEXT: arith.maxnumf

// A padded row max clamps with min only, constants high in the block and -inf outside.
// MAX: arith.constant dense<"0xFF7FFF7FFF7F{{.*}}80FF80FF
// MAX: arith.minnumf %in, %in_
// MAX-NOT: arith.maxnumf %
// MAX: arith.maxnumf %out, %in

// A padded row min clamps with max only, constants low (0xFFFF) in the block and +inf (0x7F80) outside.
// MIN: arith.constant dense<"0xFFFFFFFFFFFF{{.*}}807F807F
// MIN: arith.maxnumf %in, %in_
// MIN-NOT: arith.minnumf %in, %in_
// MIN: arith.minnumf %out, %in

// A block of whole tiles has no padded lane, so no clamp.
// WHOLE-NOT: arith.minnumf %in, %in_
// WHOLE-NOT: padding_value

// A transposed weight, x @ tl.trans(w): the load is packed once, with no transposing copy left (the pair of the
// design is composed before packing, by folding the copy into the matmul's indexing map).
// MLP-COUNT-2: linalg.pack {{.*}}inner_dims_pos = [0, 1]
// MLP-NOT: inner_dims_pos = [1, 0]

// Step 4b (ttatlas-fold-pack-pairs): the padded softmax keeps one pack per load and one unpack per store.
// FOLD-COUNT-1: linalg.pack
// FOLD-NOT: linalg.pack
// FOLD-COUNT-1: linalg.unpack
// FOLD-NOT: linalg.unpack

// Step 4c (ttatlas-tile-grid) on a 64x64x64 matmul: 2x2 output tiles, each a K chain of 2, and no loop left.
// TILE-NOT: scf.for
// TILE-COUNT-8: "reduction"]
// TILE-NOT: "reduction"]

// Step 4d, through the whole pass 4 pipeline: compute is 2-D (3 loops for a contraction), and the packs stay with
// collapse_shape between them and the compute.
// UNITS: tensor.collapse_shape
// UNITS: iterator_types = ["parallel", "parallel", "reduction"]
// UNITS-NOT: tensor<1x1x32x32xf8E4M3FN>, tensor<1x1x32x32
