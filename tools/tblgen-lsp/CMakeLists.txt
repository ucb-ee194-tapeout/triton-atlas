# The language server for .td files, on any LLVM. Its sources are LLVM's own, at our pin (llvm-project
# mlir/tools/tblgen-lsp-server and mlir/lib/Tools/tblgen-lsp-server), because an install may have neither the program nor
# its library. scripts/build.sh fetches them; nothing is copied into this repo. The target has its own name because an
# LLVM that has the program exports tblgen-lsp-server as a target.
cmake_minimum_required(VERSION 3.22)
project(tblgen_lsp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
find_package(MLIR REQUIRED CONFIG)

set(_server "${LLVM_PROJECT_DIR}/mlir/lib/Tools/tblgen-lsp-server")
add_executable(
    triton-atlas-tblgen-lsp
    "${LLVM_PROJECT_DIR}/mlir/tools/tblgen-lsp-server/tblgen-lsp-server.cpp"
    "${_server}/LSPServer.cpp"
    "${_server}/TableGenLspServerMain.cpp"
    "${_server}/TableGenServer.cpp"
)
set_target_properties(triton-atlas-tblgen-lsp PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
target_include_directories(triton-atlas-tblgen-lsp PRIVATE ${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS} "${_server}")
target_compile_options(triton-atlas-tblgen-lsp PRIVATE -fno-rtti -fno-exceptions)
target_link_libraries(
    triton-atlas-tblgen-lsp
    PRIVATE MLIRLspServerSupportLib MLIRSupport LLVMTableGen LLVMTableGenCommon LLVMSupportLSP
)
