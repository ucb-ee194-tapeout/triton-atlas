# Generate Passes.h.inc from Passes.td (as triton v3.9.0 third_party/amd/include/TritonAMDGPUToLLVM/CMakeLists.txt).
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls --name TritonToAtlasVirtual)
add_public_tablegen_target(TritonToAtlasVirtualPassIncGen)
