// RUN: not triton-atlas-opt %S/../inputs/stage3/colbroadcast_of_rowsum.mlir --ttatlas-legalize-dtypes --canonicalize --cse --ttatlas-legalize-shapes --ttatlas-assign-slots 2>&1 | FileCheck %s

// A row sum gives one value per row; broadcasting it along the rows of the tile needs it as one value per column. A
// computed vector has one layout, so pass 6 rejects the kernel by name (a loaded vector is read twice instead:
// bias_both_layouts in slots.mlir).
// CHECK: error: atlas: unsupported linalg.generic: a computed vector needed both as rows and as columns
