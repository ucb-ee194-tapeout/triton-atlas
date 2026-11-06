// Two tiles share one TTIR load. The parent pointer starts one host row in.
module attributes {ttatlas.layout = #ttatlas<layout row_major>, ttatlas.legalized = [{pass = "ttatlas-legalize-dtypes", what = "f32 output to bf16", loc = "kernel.py:12:3"}]} {
  tt.func @sliced_copy(%a: !tt.ptr<bf16>, %b: !tt.ptr<bf16>) {
    %r = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
    %c = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
    %rows = tt.expand_dims %r {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %cols = tt.expand_dims %c {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %stride = arith.constant dense<64> : tensor<32x1xi32>
    %scaled = arith.muli %rows, %stride : tensor<32x1xi32>
    %rr = tt.broadcast %scaled : tensor<32x1xi32> -> tensor<32x64xi32>
    %cc = tt.broadcast %cols : tensor<1x64xi32> -> tensor<32x64xi32>
    %offsets = arith.addi %rr, %cc : tensor<32x64xi32>
    %parent_offset = arith.constant 64 : i32
    %a1 = tt.addptr %a, %parent_offset : !tt.ptr<bf16>, i32
    %base = tt.splat %a1 : !tt.ptr<bf16> -> tensor<32x64x!tt.ptr<bf16>>
    %p = tt.addptr %base, %offsets : tensor<32x64x!tt.ptr<bf16>>, tensor<32x64xi32>
    %x = tt.load %p {ttatlas.host_dtype = bf16} : tensor<32x64x!tt.ptr<bf16>>
    %left = tensor.extract_slice %x[0, 0] [32, 32] [1, 1] : tensor<32x64xbf16> to tensor<32x32xbf16>
    %right = tensor.extract_slice %x[0, 32] [32, 32] [1, 1] : tensor<32x64xbf16> to tensor<32x32xbf16>
    %again = tensor.extract_slice %x[0, 0] [32, 32] [1, 1] : tensor<32x64xbf16> to tensor<32x32xbf16>
    %out = tensor.empty() : tensor<32x64xbf16>
    %o0 = tensor.insert_slice %again into %out[0, 0] [32, 32] [1, 1] : tensor<32x32xbf16> into tensor<32x64xbf16>
    %o1 = tensor.insert_slice %right into %o0[0, 32] [32, 32] [1, 1] : tensor<32x32xbf16> into tensor<32x64xbf16>
    %dest = tt.splat %b : !tt.ptr<bf16> -> tensor<32x64x!tt.ptr<bf16>>
    %q = tt.addptr %dest, %offsets : tensor<32x64x!tt.ptr<bf16>>, tensor<32x64xi32>
    tt.store %q, %o1 {ttatlas.host_dtype = f32} : tensor<32x64x!tt.ptr<bf16>>
    tt.return
  }
}
