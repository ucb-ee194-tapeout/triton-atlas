// RUN: triton-atlas-opt %s --split-input-file --verify-diagnostics --ttatlas-verify-contract=stage=7

// Pass 7's contract: no TTIR is left; one argument-free func.func carrying ttatlas.kernel.

func.func @ok() -> !atlas.virtual_state attributes {ttatlas.kernel = "{\22version\22: 1}"} {
  %io0 = "atlas.virtual_start"() : () -> !atlas.virtual_state
  return %io0 : !atlas.virtual_state
}

// -----

// expected-error @below {{contract of pass 7 broken: func.func has arguments}}
func.func @args(%x: i32) attributes {ttatlas.kernel = "{\22version\22: 1}"} {
  return
}

// -----

// expected-error @below {{contract of pass 7 broken: func.func without ttatlas.kernel}}
func.func @no_meta() {
  return
}

// -----

// expected-error @below {{contract of pass 7 broken: expected one func.func, found 0}}
module {
  // expected-error @below {{contract of pass 7 broken: TTIR is left: tt.func}}
  tt.func @k() {
    // expected-error @below {{contract of pass 7 broken: TTIR is left: tt.return}}
    tt.return
  }
}

// -----

// expected-error @below {{contract of pass 7 broken: expected one func.func, found 2}}
module {
  func.func @a() attributes {ttatlas.kernel = "{\22version\22: 1}"} {
    return
  }
  func.func @b() attributes {ttatlas.kernel = "{\22version\22: 1}"} {
    return
  }
}

// -----

// A func.func inside another op is not the kernel, even when the count of top-level ones is right.
module {
  func.func @k() attributes {ttatlas.kernel = "{\22version\22: 1}"} {
    return
  }
  builtin.module {
    // expected-error @below {{contract of pass 7 broken: func.func nested inside another op}}
    func.func @inner() attributes {ttatlas.kernel = "{\22version\22: 1}"} {
      return
    }
  }
}
