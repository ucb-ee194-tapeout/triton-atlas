# The language server, linked to the wheel's libtriton. It isn't installed into the wheel. libtriton is a Python
# extension, so a plain program must also load libpython.
# Ask for the file by name: conda's Python reports its static library even when the shared one is there.
execute_process(
    COMMAND
        ${Python_EXECUTABLE} -c
        "import os, sysconfig as s; print(os.path.join(s.get_config_var('LIBDIR'), 'libpython' + s.get_config_var('LDVERSION') + '.so'))"
    OUTPUT_VARIABLE _python_lib
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT EXISTS "${_python_lib}")
    message(STATUS "No shared libpython at ${_python_lib}; skipping triton-atlas-lsp")
    return()
endif()

add_executable(triton-atlas-lsp triton-atlas-lsp.cpp)
add_dependencies(triton-atlas-lsp MLIRAtlas)
set_target_properties(
    triton-atlas-lsp
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin" BUILD_RPATH "${TRITON_WHEEL_DIR}/_C"
)
target_include_directories(
    triton-atlas-lsp
    SYSTEM
    PRIVATE ${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS} "${TRITON_WHEEL_DIR}/include"
)
target_include_directories(
    triton-atlas-lsp
    PRIVATE ${PROJECT_SOURCE_DIR}/third_party/atlas-mlir/include ${PROJECT_BINARY_DIR}/atlas-mlir/include
)
target_compile_options(triton-atlas-lsp PRIVATE -fvisibility=hidden -fno-rtti -fno-exceptions)
# As with the plugin, MLIR and LLVM come from libtriton; these static libraries hold only what it doesn't export.
target_link_libraries(
    triton-atlas-lsp
    PRIVATE
        $<TARGET_FILE:MLIRAtlas>
        "${LLVM_LIBRARY_DIR}/libMLIRLspServerLib.a"
        "${LLVM_LIBRARY_DIR}/libMLIRLspServerSupportLib.a"
        "${LLVM_LIBRARY_DIR}/libLLVMSupportLSP.a"
        "${LLVM_LIBRARY_DIR}/libMLIRBytecodeWriter.a"
        "${LLVM_LIBRARY_DIR}/libMLIRBytecodeOpInterface.a"
        "${TRITON_LIB}"
        "${_python_lib}"
        -Wl,--allow-shlib-undefined
)

# Editors use this stable path. scripts/build-lsp-from-source.sh repoints it, so leave an existing link alone.
if(NOT EXISTS "${CMAKE_BINARY_DIR}/../triton-atlas-lsp")
    file(CREATE_LINK "${CMAKE_BINARY_DIR}/bin/triton-atlas-lsp" "${CMAKE_BINARY_DIR}/../triton-atlas-lsp" SYMBOLIC)
endif()
