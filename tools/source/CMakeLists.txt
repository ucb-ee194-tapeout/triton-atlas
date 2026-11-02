# The language server built from the Triton submodule, with no wheel or libpython involved. A wrapper project, so
# Triton stays unpatched: it builds Triton and atlas-mlir side by side, then links the server to both.
cmake_minimum_required(VERSION 3.22)
project(triton_atlas_lsp LANGUAGES C CXX)

# Triton finds MLIR in its own scope, so find it here too for the server's headers.
find_package(MLIR REQUIRED CONFIG)

get_filename_component(_repo "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
# Triton's setup.py normally passes this; the source names its own version.
file(STRINGS "${_repo}/third_party/triton/python/triton/__init__.py" _triton_source REGEX "^__version__ = ")
string(REGEX MATCH "'([^'+]+)" _ "${_triton_source}")
set(TRITON_VERSION "${CMAKE_MATCH_1}" CACHE STRING "Triton version")

add_subdirectory("${_repo}/third_party/triton" triton)
get_property(_triton_libs GLOBAL PROPERTY TRITON_LIBS)
add_subdirectory("${_repo}/third_party/atlas-mlir" atlas-mlir EXCLUDE_FROM_ALL)

add_executable(triton-atlas-lsp EXCLUDE_FROM_ALL "${_repo}/tools/triton-atlas-lsp.cpp")
set_target_properties(triton-atlas-lsp PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
target_compile_definitions(triton-atlas-lsp PRIVATE TRITON_ATLAS_LSP_FROM_SOURCE)
target_compile_options(triton-atlas-lsp PRIVATE -fno-rtti -fno-exceptions)
target_include_directories(
    triton-atlas-lsp
    PRIVATE
        ${MLIR_INCLUDE_DIRS}
        ${LLVM_INCLUDE_DIRS}
        "${_repo}/third_party/triton/include"
        "${CMAKE_BINARY_DIR}/triton/include"
        "${_repo}/third_party/atlas-mlir/include"
        "${CMAKE_BINARY_DIR}/atlas-mlir/include"
)
add_dependencies(triton-atlas-lsp TritonTableGen TritonGPUTableGen)
target_link_libraries(triton-atlas-lsp PRIVATE MLIRAtlas ${_triton_libs} MLIRLspServerLib MLIRRegisterAllDialects)
