// Fixture: Triton tutorial 03 style matmul, 64x64 blocks, runtime strides and masks, a 2-D grid. Outside the
// accepted subset today (ttatlas-validate-ttir stops it; that lane adds the test showing it, in
// test/validate-ttir/); kept as the target for specialization and tiling.
// TTIR from Triton 3.9.0 (A, Bm: *fp8e4nv, C: *bf16, M N K and strides: i32, BM = BN = 64, BK = 32), debug
// locations dropped.
//   @triton.jit
//   def mm(A, Bm, C, M, N, K, sam, sak, sbk, sbn, scm, scn, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
//       pm = tl.program_id(0); pn = tl.program_id(1)
//       rm = pm * BM + tl.arange(0, BM); rn = pn * BN + tl.arange(0, BN); rk = tl.arange(0, BK)
//       a = A + rm[:, None] * sam + rk[None, :] * sak; b = Bm + rk[:, None] * sbk + rn[None, :] * sbn
//       acc = tl.zeros((BM, BN), tl.float32)
//       for k in range(0, tl.cdiv(K, BK)):
//           av = tl.load(a, mask=rk[None, :] < K - k * BK, other=0.0)
//           bv = tl.load(b, mask=rk[:, None] < K - k * BK, other=0.0)
//           acc = tl.dot(av, bv, acc); a += BK * sak; b += BK * sbk
//       c = C + rm[:, None] * scm + rn[None, :] * scn
//       tl.store(c, acc.to(tl.bfloat16), mask=(rm[:, None] < M) & (rn[None, :] < N))

module {
  tt.func public @mm(%arg0: !tt.ptr<f8E4M3FN>, %arg1: !tt.ptr<f8E4M3FN>, %arg2: !tt.ptr<bf16>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32) attributes {noinline = false} {
    %c31_i32 = arith.constant 31 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<32x64xf8E4M3FN>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<64x32xf8E4M3FN>
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c64_i32 = arith.constant 64 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = arith.muli %0, %c64_i32 : i32
    %3 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %4 = tt.splat %2 : i32 -> tensor<64xi32>
    %5 = arith.addi %4, %3 : tensor<64xi32>
    %6 = arith.muli %1, %c64_i32 : i32
    %7 = tt.splat %6 : i32 -> tensor<64xi32>
    %8 = arith.addi %7, %3 : tensor<64xi32>
    %9 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %10 = tt.expand_dims %5 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %11 = tt.splat %arg6 : i32 -> tensor<64x1xi32>
    %12 = arith.muli %10, %11 : tensor<64x1xi32>
    %13 = tt.splat %arg0 : !tt.ptr<f8E4M3FN> -> tensor<64x1x!tt.ptr<f8E4M3FN>>
    %14 = tt.addptr %13, %12 : tensor<64x1x!tt.ptr<f8E4M3FN>>, tensor<64x1xi32>
    %15 = tt.expand_dims %9 {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %16 = tt.splat %arg7 : i32 -> tensor<1x32xi32>
    %17 = arith.muli %15, %16 : tensor<1x32xi32>
    %18 = tt.broadcast %14 : tensor<64x1x!tt.ptr<f8E4M3FN>> -> tensor<64x32x!tt.ptr<f8E4M3FN>>
    %19 = tt.broadcast %17 : tensor<1x32xi32> -> tensor<64x32xi32>
    %20 = tt.addptr %18, %19 : tensor<64x32x!tt.ptr<f8E4M3FN>>, tensor<64x32xi32>
    %21 = tt.expand_dims %9 {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %22 = tt.splat %arg8 : i32 -> tensor<32x1xi32>
    %23 = arith.muli %21, %22 : tensor<32x1xi32>
    %24 = tt.splat %arg1 : !tt.ptr<f8E4M3FN> -> tensor<32x1x!tt.ptr<f8E4M3FN>>
    %25 = tt.addptr %24, %23 : tensor<32x1x!tt.ptr<f8E4M3FN>>, tensor<32x1xi32>
    %26 = tt.expand_dims %8 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %27 = tt.splat %arg9 : i32 -> tensor<1x64xi32>
    %28 = arith.muli %26, %27 : tensor<1x64xi32>
    %29 = tt.broadcast %25 : tensor<32x1x!tt.ptr<f8E4M3FN>> -> tensor<32x64x!tt.ptr<f8E4M3FN>>
    %30 = tt.broadcast %28 : tensor<1x64xi32> -> tensor<32x64xi32>
    %31 = tt.addptr %29, %30 : tensor<32x64x!tt.ptr<f8E4M3FN>>, tensor<32x64xi32>
    %32 = arith.addi %arg5, %c31_i32 : i32
    %33 = arith.divsi %32, %c32_i32 : i32
    %34:3 = scf.for %arg12 = %c0_i32 to %33 step %c1_i32 iter_args(%arg13 = %20, %arg14 = %31, %arg15 = %cst) -> (tensor<64x32x!tt.ptr<f8E4M3FN>>, tensor<32x64x!tt.ptr<f8E4M3FN>>, tensor<64x64xf32>)  : i32 {
      %52 = arith.muli %arg12, %c32_i32 : i32
      %53 = arith.subi %arg5, %52 : i32
      %54 = tt.splat %53 : i32 -> tensor<1x32xi32>
      %55 = arith.cmpi slt, %15, %54 : tensor<1x32xi32>
      %56 = tt.broadcast %55 : tensor<1x32xi1> -> tensor<64x32xi1>
      %57 = tt.load %arg13, %56, %cst_1 : tensor<64x32x!tt.ptr<f8E4M3FN>>
      %58 = tt.splat %53 : i32 -> tensor<32x1xi32>
      %59 = arith.cmpi slt, %21, %58 : tensor<32x1xi32>
      %60 = tt.broadcast %59 : tensor<32x1xi1> -> tensor<32x64xi1>
      %61 = tt.load %arg14, %60, %cst_0 : tensor<32x64x!tt.ptr<f8E4M3FN>>
      %62 = tt.dot %57, %61, %arg15 : tensor<64x32xf8E4M3FN> * tensor<32x64xf8E4M3FN> -> tensor<64x64xf32>
      %63 = arith.muli %arg7, %c32_i32 : i32
      %64 = tt.splat %63 : i32 -> tensor<64x32xi32>
      %65 = tt.addptr %arg13, %64 : tensor<64x32x!tt.ptr<f8E4M3FN>>, tensor<64x32xi32>
      %66 = arith.muli %arg8, %c32_i32 : i32
      %67 = tt.splat %66 : i32 -> tensor<32x64xi32>
      %68 = tt.addptr %arg14, %67 : tensor<32x64x!tt.ptr<f8E4M3FN>>, tensor<32x64xi32>
      scf.yield %65, %68, %62 : tensor<64x32x!tt.ptr<f8E4M3FN>>, tensor<32x64x!tt.ptr<f8E4M3FN>>, tensor<64x64xf32>
    }
    %35 = tt.splat %arg10 : i32 -> tensor<64x1xi32>
    %36 = arith.muli %10, %35 : tensor<64x1xi32>
    %37 = tt.splat %arg2 : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
    %38 = tt.addptr %37, %36 : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi32>
    %39 = tt.splat %arg11 : i32 -> tensor<1x64xi32>
    %40 = arith.muli %26, %39 : tensor<1x64xi32>
    %41 = tt.broadcast %38 : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x64x!tt.ptr<bf16>>
    %42 = tt.broadcast %40 : tensor<1x64xi32> -> tensor<64x64xi32>
    %43 = tt.addptr %41, %42 : tensor<64x64x!tt.ptr<bf16>>, tensor<64x64xi32>
    %44 = tt.splat %arg3 : i32 -> tensor<64x1xi32>
    %45 = arith.cmpi slt, %10, %44 : tensor<64x1xi32>
    %46 = tt.splat %arg4 : i32 -> tensor<1x64xi32>
    %47 = arith.cmpi slt, %26, %46 : tensor<1x64xi32>
    %48 = tt.broadcast %45 : tensor<64x1xi1> -> tensor<64x64xi1>
    %49 = tt.broadcast %47 : tensor<1x64xi1> -> tensor<64x64xi1>
    %50 = arith.andi %48, %49 : tensor<64x64xi1>
    %51 = arith.truncf %34#2 : tensor<64x64xf32> to tensor<64x64xbf16>
    tt.store %43, %51, %50 : tensor<64x64x!tt.ptr<bf16>>
    tt.return
  }
}
