"""Prints a compute-free kernel with 70 tiles in and out: 32 words each (16 in, 1 move, 15 out), over atlas-mlir's 2048-word limit."""
N = 70


def slot(index, arg, role, host):
    return (f"#ttatlas.slot<index = {index}, kind = memory, arg = {arg}, role = {role}, offset = {index * 1024}, "
            f"row_stride = 32, col_stride = 1, block = [0, 0, 32, 32], valid = [0, 0, 32, 32], vector_layout = none, "
            f"transposed = false, host_dtype = {host}, tile_dtype = bf16>")


print("module {")
print("  // expected-error @below {{kernel too large for the pinned instruction memory}}")
print("  tt.func @too_large(%a: !tt.ptr<bf16>, %b: !tt.ptr<bf16>) {")
print("    %p = tt.splat %a : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>")
print("    %q = tt.splat %b : !tt.ptr<bf16> -> tensor<32x32x!tt.ptr<bf16>>")
for i in range(N):
    print(
        f"    %v{i} = tt.load %p {{ttatlas.slot = {slot(i, 0, 'constant_bf16', 'bf16')}}} : tensor<32x32x!tt.ptr<bf16>>"
    )
    print(
        f"    tt.store %q, %v{i} {{ttatlas.slot = {slot(i, 1, 'output_bf16', 'bf16')}}} : tensor<32x32x!tt.ptr<bf16>>")
print("    tt.return")
print("  }")
print("}")
