// RUN: triton-atlas-opt %s --split-input-file --convert-triton-compute-to-linalg --ttatlas-verify-contract=stage=3 | FileCheck %s

// Broadcast group: float tt.broadcast becomes one linalg.broadcast, tt.splat of a float becomes linalg.fill, and an
// expand_dims no broadcast absorbs becomes tensor.expand_shape. Broadcasts of offsets stay TTIR.

// The bias of the MLP: expand_dims + broadcast of a 1-D row become one linalg.broadcast over dimension 0.
// CHECK-LABEL: tt.func @bias_row
// CHECK-NOT: tt.expand_dims
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<32x32xf32>
// CHECK: %[[B:.*]] = linalg.broadcast ins(%{{.*}} : tensor<32xf32>) outs(%[[EMPTY]] : tensor<32x32xf32>) dimensions = [0]
// CHECK-NEXT: tt.return %[[B]]
tt.func @bias_row(%b: tensor<32xf32>) -> tensor<32x32xf32> {
  %e = tt.expand_dims %b {axis = 0 : i32} : tensor<32xf32> -> tensor<1x32xf32>
  %r = tt.broadcast %e : tensor<1x32xf32> -> tensor<32x32xf32>
  tt.return %r : tensor<32x32xf32>
}

// -----

// A column (32x1) with no expand_dims under it: the unit dimension is collapsed away, then broadcast.
// CHECK-LABEL: tt.func @column
// CHECK: %[[C:.*]] = tensor.collapse_shape %{{.*}} {{\[\[}}0, 1]] : tensor<32x1xbf16> into tensor<32xbf16>
// CHECK: linalg.broadcast ins(%[[C]] : tensor<32xbf16>) outs(%{{.*}} : tensor<32x32xbf16>) dimensions = [1]
tt.func @column(%x: tensor<32x1xbf16>) -> tensor<32x32xbf16> {
  %r = tt.broadcast %x : tensor<32x1xbf16> -> tensor<32x32xbf16>
  tt.return %r : tensor<32x32xbf16>
}

// -----

// CHECK-LABEL: tt.func @expand_only
// CHECK: tensor.expand_shape %{{.*}} {{\[\[}}0, 1]] output_shape [1, 32] : tensor<32xbf16> into tensor<1x32xbf16>
tt.func @expand_only(%x: tensor<32xbf16>) -> tensor<1x32xbf16> {
  %r = tt.expand_dims %x {axis = 0 : i32} : tensor<32xbf16> -> tensor<1x32xbf16>
  tt.return %r : tensor<1x32xbf16>
}

// -----

// CHECK-LABEL: tt.func @splat
// CHECK: linalg.fill ins(%{{.*}} : bf16) outs(%{{.*}} : tensor<32x32xbf16>)
tt.func @splat(%s: bf16) -> tensor<32x32xbf16> {
  %r = tt.splat %s : bf16 -> tensor<32x32xbf16>
  tt.return %r : tensor<32x32xbf16>
}

// -----

// Offsets are address math: their broadcasts stay TTIR.
// CHECK-LABEL: tt.func @offsets
// CHECK: tt.expand_dims
// CHECK: tt.broadcast
// CHECK-NOT: linalg
tt.func @offsets(%r: tensor<32xi32>) -> tensor<32x32xi32> {
  %e = tt.expand_dims %r {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
  %b = tt.broadcast %e : tensor<1x32xi32> -> tensor<32x32xi32>
  tt.return %b : tensor<32x32xi32>
}

// -----

// An expand_dims the broadcast doesn't absorb (it stretches another dimension) becomes tensor.expand_shape; no float
// tt.expand_dims is left.
// CHECK-LABEL: tt.func @not_absorbed
// CHECK-NOT: tt.expand_dims
// CHECK: tensor.expand_shape
// CHECK: linalg.broadcast {{.*}} dimensions = [1]
tt.func @not_absorbed(%x: tensor<1x32xbf16>) -> tensor<1x4x32xbf16> {
  %e = tt.expand_dims %x {axis = 0 : i32} : tensor<1x32xbf16> -> tensor<1x1x32xbf16>
  %r = tt.broadcast %e : tensor<1x1x32xbf16> -> tensor<1x4x32xbf16>
  tt.return %r : tensor<1x4x32xbf16>
}

// -----

// What the conversion leaves dead is erased with MLIR's region DCE, chains included (a constant whose only user was a
// dead expand_dims), while the kernel's arguments stay, even unused ones such as Inductor's size arguments.
// CHECK-LABEL: tt.func public @dead_chain(%arg0: tensor<32xf32>, %arg1: i32, %arg2: f32)
// CHECK-NOT: arith.constant dense<2.0
// CHECK-NOT: tt.expand_dims
// CHECK: linalg.broadcast
// CHECK: tt.return
tt.func public @dead_chain(%bias: tensor<32xf32>, %xnumel: i32, %unused: f32) -> tensor<32x32xf32> {
  %c = arith.constant dense<2.000000e+00> : tensor<32xf32>
  %dead = tt.expand_dims %c {axis = 0 : i32} : tensor<32xf32> -> tensor<1x32xf32>
  %e = tt.expand_dims %bias {axis = 0 : i32} : tensor<32xf32> -> tensor<1x32xf32>
  %b = tt.broadcast %e : tensor<1x32xf32> -> tensor<32x32xf32>
  tt.return %b : tensor<32x32xf32>
}
