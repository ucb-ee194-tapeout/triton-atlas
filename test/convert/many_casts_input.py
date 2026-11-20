"""Prints a compute-free kernel with 30 tiles, each loaded, cast to f32 and stored."""
N = 30


def slot(index, arg, role, host):
    return (f"#ttatlas.slot<index = {index}, kind = memory, arg = {arg}, role = {role}, offset = {index * 1024}, "
            f"row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, "
            f"transposed = false, host_dtype = {host}, tile_dtype = bf16>")


print("module {")
print("  tt.func @many_casts(%a: !tt.ptr<bf16>, %b: !tt.ptr<f32>) {")
print("    %p = tt.splat %a : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>")
print("    %q = tt.splat %b : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>>")
for i in range(N):
    print(
        f"    %v{i} = tt.load %p {{ttatlas.slot = {slot(i, 0, 'constant_bf16', 'bf16')}}} : tensor<32x32x!tt.ptr<bf16>>"
    )
    print(f"    %w{i} = arith.extf %v{i} : tensor<32x32xbf16> to tensor<32x32xf32>")
    print(f"    tt.store %q, %w{i} {{ttatlas.slot = {slot(i, 1, 'output_bf16', 'f32')}}} : tensor<32x32x!tt.ptr<f32>>")
print("    tt.return")
print("  }")
print("}")
