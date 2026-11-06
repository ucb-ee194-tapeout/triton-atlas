// RUN: triton-atlas-opt %s --verify-diagnostics --ttatlas-verify-contract=stage=9

// expected-error @below {{ttatlas-verify-contract needs stage=1..7, got 9}}
module {}
