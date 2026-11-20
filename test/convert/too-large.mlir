// RUN: %PYTHON %S/too_large_input.py | triton-atlas-opt --convert-triton-to-atlas-virtual --verify-diagnostics

// A kernel whose program cannot fit atlas-mlir's instruction memory (2048 words at our pin) is rejected with that reason
// instead of failing later in lowering. The size is a lower bound (16 words per BF16 input, 11 per FP8 input, 15 per
// output, one per other op), so a kernel that fails here would fail in atlas-mlir too. The 70 tiles of too_large_input.py need about 2200 words.
