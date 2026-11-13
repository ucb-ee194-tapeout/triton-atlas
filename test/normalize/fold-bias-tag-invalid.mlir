// RUN: triton-atlas-opt %s --verify-diagnostics

// The module tag must be a bool (the ttatlas dialect's tag check), so a typo such as a string is caught when the IR is
// read, not silently treated as off.

// expected-error @below {{ttatlas.fold_bias_into_dot must be a bool}}
module attributes {ttatlas.fold_bias_into_dot = "true"} {
}
