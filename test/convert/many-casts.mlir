// RUN: %PYTHON %S/many_casts_input.py | triton-atlas-opt --convert-triton-to-atlas-virtual --verify-atlas-virtual-stream | grep -c virtual_output_bf16 | FileCheck %s

// Thirty load, cast, store tiles in one kernel: each cast gives its result the tile of its operand, which must stay
// valid while the map of tiles grows. Every tile is read and written once.

// CHECK: 30
