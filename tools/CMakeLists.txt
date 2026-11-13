# Two programs linked to the wheel's libtriton, which aren't in the wheel: the small language server (make
# build.lsp.small) and triton-atlas-opt (make build.tools).
# libtriton is a Python extension, so a plain program must also load libpython.
# Ask for the file by name: conda's Python reports its static library even when the shared one is there.
execute_process(
    COMMAND
        ${Python_EXECUTABLE} -c
        "import os, sysconfig as s; print(os.path.join(s.get_config_var('LIBDIR'), 'libpython' + s.get_config_var('LDVERSION') + '.so'))"
    OUTPUT_VARIABLE _python_lib
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT EXISTS "${_python_lib}")
    add_custom_target(
        triton-atlas-lsp
        COMMAND ${CMAKE_COMMAND} -E echo "No shared libpython at ${_python_lib}; try make build.lsp.full"
        COMMAND ${CMAKE_COMMAND} -E false
    )
    add_custom_target(
        triton-atlas-opt
        COMMAND ${CMAKE_COMMAND} -E echo "No shared libpython at ${_python_lib}"
        COMMAND ${CMAKE_COMMAND} -E false
    )
    return()
endif()

# Both register every upstream MLIR dialect. libtriton comes first on the link line, so whatever it exports comes from
# it; MLIR's static libraries then fill in only what it lacks. A second copy of what libtriton exports would break
# MLIR's type registry (see the top CMakeLists.txt).
file(GLOB _mlir_archives "${LLVM_LIBRARY_DIR}/libMLIR*.a" "${LLVM_LIBRARY_DIR}/libLLVM*.a")

# The language server.
add_executable(triton-atlas-lsp EXCLUDE_FROM_ALL triton-atlas-lsp.cpp)
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
target_link_libraries(
    triton-atlas-lsp
    PRIVATE
        "${TRITON_LIB}"
        $<TARGET_FILE:MLIRAtlas>
        "${LLVM_LIBRARY_DIR}/libMLIRLspServerLib.a"
        "${LLVM_LIBRARY_DIR}/libMLIRRegisterAllDialects.a"
        -Wl,--start-group
        ${_mlir_archives}
        -Wl,--end-group
        "${_python_lib}"
        -Wl,--allow-shlib-undefined
)

# triton-atlas-opt: also every upstream MLIR pass and MLIR's opt driver. It loads our plugin at runtime, so it needs
# the plugin built, and TRITON_ATLAS_PLUGIN tells it where.
add_executable(triton-atlas-opt EXCLUDE_FROM_ALL triton-atlas-opt.cpp)
add_dependencies(triton-atlas-opt TritonAtlasBackend)
target_compile_definitions(triton-atlas-opt PRIVATE TRITON_ATLAS_PLUGIN="$<TARGET_FILE:TritonAtlasBackend>")
set_target_properties(
    triton-atlas-opt
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin" BUILD_RPATH "${TRITON_WHEEL_DIR}/_C"
)
target_include_directories(
    triton-atlas-opt
    SYSTEM
    PRIVATE ${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS} "${TRITON_WHEEL_DIR}/include"
)
target_compile_options(triton-atlas-opt PRIVATE -fvisibility=hidden -fno-rtti -fno-exceptions)
# The plugin comes right after libtriton: it carries its own linalg, and the tool must use that copy, not link a
# second one from the archives. Two copies register two linalg dialects and abort ("Trying to register different
# dialects for the same namespace: linalg").
target_link_libraries(
    triton-atlas-opt
    PRIVATE
        "${TRITON_LIB}"
        $<TARGET_FILE:TritonAtlasBackend>
        "${LLVM_LIBRARY_DIR}/libMLIROptLib.a"
        "${LLVM_LIBRARY_DIR}/libMLIRRegisterAllDialects.a"
        "${LLVM_LIBRARY_DIR}/libMLIRRegisterAllPasses.a"
        "${LLVM_LIBRARY_DIR}/libMLIRRegisterAllExtensions.a"
        -Wl,--start-group
        ${_mlir_archives}
        -Wl,--end-group
        "${_python_lib}"
        -Wl,--allow-shlib-undefined
)
