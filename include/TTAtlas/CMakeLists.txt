# Generate the ttatlas dialect's C++ from its TableGen files (same pattern as
# include/TritonToAtlasVirtual/CMakeLists.txt; mlir_tablegen as in triton v3.9.0
# include/triton/Dialect/TritonGPU/IR/CMakeLists.txt).
set(LLVM_TARGET_DEFINITIONS TTAtlasAttrs.td)
mlir_tablegen(TTAtlasDialect.h.inc -gen-dialect-decls -dialect=ttatlas)
mlir_tablegen(TTAtlasDialect.cpp.inc -gen-dialect-defs -dialect=ttatlas)
mlir_tablegen(TTAtlasAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=ttatlas)
mlir_tablegen(TTAtlasAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=ttatlas)
mlir_tablegen(TTAtlasEnums.h.inc -gen-enum-decls)
mlir_tablegen(TTAtlasEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(TTAtlasIncGen)
