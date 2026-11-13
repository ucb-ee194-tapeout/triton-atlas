// RUN: triton-atlas-opt %s --convert-triton-compute-to-linalg --ttatlas-verify-contract=stage=3 | FileCheck %s --check-prefix=DEFAULT
// RUN: triton-atlas-opt %S/../fixtures/mlp.ttir.mlir --convert-triton-compute-to-linalg --ttatlas-verify-contract=stage=3 | FileCheck %s --check-prefix=FOLDED

// The first version's kernel, the MLP (relu(x @ w.T + b), one 32x32 tile), through pass 3: all its compute is linalg
// (the bias extf, the transpose, the matmul, the bias broadcast, the bias add, the ReLU and the BF16 truncf), while
// loads, stores and address math stay TTIR.
//
// The DEFAULT checks: the input below is the MLP fixture as pass 2 leaves it by default (written by hand, since pass 2 lives on
// another branch): the dot starts from zero and the bias is a separate add. That is what the pipeline gives pass 3.
// The FOLDED checks: the raw fixture, where Triton's combine folded the bias into the dot's accumulator. Pass 2 keeps that form
// only with its experimental fold-bias-into-dot option; the bias then becomes the matmul's output operand.

// DEFAULT-LABEL: tt.func public @mlp_after_pass2
// DEFAULT: %[[X:.*]] = tt.load
// DEFAULT: %[[W:.*]] = tt.load
// DEFAULT: %[[B:.*]] = tt.load
// DEFAULT: %[[BF:.*]] = linalg.generic {{.*}} ins(%[[B]] : tensor<32xbf16>)
// DEFAULT: arith.extf
// DEFAULT: %[[WT:.*]] = linalg.transpose ins(%[[W]] : tensor<32x32xf8E4M3FN>)
// DEFAULT: %[[BB:.*]] = linalg.broadcast ins(%[[BF]] : tensor<32xf32>) {{.*}} dimensions = [0]
// DEFAULT: %[[F:.*]] = linalg.fill
// DEFAULT: %[[M:.*]] = linalg.matmul ins(%[[X]], %[[WT]] : {{.*}}) outs(%[[F]] : tensor<32x32xf32>)
// DEFAULT: %[[S:.*]] = linalg.generic {{.*}} ins(%[[M]], %[[BB]] : tensor<32x32xf32>, tensor<32x32xf32>)
// DEFAULT: arith.addf
// DEFAULT: %[[R:.*]] = linalg.generic {{.*}} ins(%[[S]], %{{.*}} : tensor<32x32xf32>, tensor<32x32xf32>)
// DEFAULT: arith.maxnumf
// DEFAULT: %[[T:.*]] = linalg.generic {{.*}} ins(%[[R]] : tensor<32x32xf32>)
// DEFAULT: arith.truncf
// DEFAULT: tt.store %{{.*}}, %[[T]]

// FOLDED-LABEL: tt.func public @mlp
// FOLDED: %[[X:.*]] = tt.load
// FOLDED: %[[W:.*]] = tt.load
// FOLDED: %[[B:.*]] = tt.load
// FOLDED: %[[BF:.*]] = linalg.generic {{.*}} ins(%[[B]] : tensor<32xbf16>)
// FOLDED: arith.extf
// FOLDED: %[[WT:.*]] = linalg.transpose ins(%[[W]] : tensor<32x32xf8E4M3FN>)
// FOLDED: %[[BB:.*]] = linalg.broadcast ins(%[[BF]] : tensor<32xf32>) {{.*}} dimensions = [0]
// FOLDED-NOT: linalg.fill
// FOLDED: %[[M:.*]] = linalg.matmul ins(%[[X]], %[[WT]] : {{.*}}) outs(%[[BB]] : tensor<32x32xf32>)
// FOLDED: %[[R:.*]] = linalg.generic {{.*}} ins(%[[M]], %{{.*}} : tensor<32x32xf32>, tensor<32x32xf32>)
// FOLDED: arith.maxnumf
// FOLDED: %[[T:.*]] = linalg.generic {{.*}} ins(%[[R]] : tensor<32x32xf32>)
// FOLDED: arith.truncf
// FOLDED: tt.store %{{.*}}, %[[T]]

module {
  tt.func public @mlp_after_pass2(%arg0: !tt.ptr<f8E4M3FN>, %arg1: !tt.ptr<f8E4M3FN>, %arg2: !tt.ptr<bf16>, %arg3: !tt.ptr<bf16>) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
    %cst_0 = arith.constant dense<32> : tensor<32x1xi32>
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %2 = arith.muli %1, %cst_0 : tensor<32x1xi32>
    %3 = tt.expand_dims %0 {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %4 = tt.broadcast %2 : tensor<32x1xi32> -> tensor<32x32xi32>
    %5 = tt.broadcast %3 : tensor<1x32xi32> -> tensor<32x32xi32>
    %6 = arith.addi %4, %5 : tensor<32x32xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f8E4M3FN> -> tensor<32x32x!tt.ptr<f8E4M3FN>>
    %8 = tt.addptr %7, %6 : tensor<32x32x!tt.ptr<f8E4M3FN>>, tensor<32x32xi32>
    %9 = tt.load %8 : tensor<32x32x!tt.ptr<f8E4M3FN>>
    %10 = tt.splat %arg1 : !tt.ptr<f8E4M3FN> -> tensor<32x32x!tt.ptr<f8E4M3FN>>
    %11 = tt.addptr %10, %6 : tensor<32x32x!tt.ptr<f8E4M3FN>>, tensor<32x32xi32>
    %12 = tt.load %11 : tensor<32x32x!tt.ptr<f8E4M3FN>>
    %13 = tt.splat %arg2 : !tt.ptr<bf16> -> tensor<32x!tt.ptr<bf16>>
    %14 = tt.addptr %13, %0 : tensor<32x!tt.ptr<bf16>>, tensor<32xi32>
    %15 = tt.load %14 : tensor<32x!tt.ptr<bf16>>
    %16 = arith.extf %15 : tensor<32xbf16> to tensor<32xf32>
    %17 = tt.trans %12 {order = array<i32: 1, 0>} : tensor<32x32xf8E4M3FN> -> tensor<32x32xf8E4M3FN>
    %18 = tt.expand_dims %16 {axis = 0 : i32} : tensor<32xf32> -> tensor<1x32xf32>
    %19 = tt.broadcast %18 : tensor<1x32xf32> -> tensor<32x32xf32>
    %zero = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
    %d = tt.dot %9, %17, %zero : tensor<32x32xf8E4M3FN> * tensor<32x32xf8E4M3FN> -> tensor<32x32xf32>
    %20 = arith.addf %d, %19 : tensor<32x32xf32>
    %21 = tt.splat %arg3 : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>
    %22 = tt.addptr %21, %6 : tensor<32x32x!tt.ptr<bf16>>, tensor<32x32xi32>
    %23 = arith.maxnumf %20, %cst : tensor<32x32xf32>
    %24 = arith.truncf %23 : tensor<32x32xf32> to tensor<32x32xbf16>
    tt.store %22, %24 : tensor<32x32x!tt.ptr<bf16>>
    tt.return
  }
}
