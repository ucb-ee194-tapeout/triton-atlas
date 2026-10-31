# Builds the Atlas backend's plugin: one shared library that the stock Triton wheel loads at runtime.
# Recipe from triton-lang/triton-ext@268252b cmake/triton-ext.cmake and cmake/llvm-abi-breaking.cmake (MIT):
# compile against the wheel's headers and an LLVM install of the same commit, and link only against libtriton, which
# exports MLIR and LLVM. A second copy of MLIR in our library would break MLIR's type registry.
cmake_minimum_required(VERSION 3.22)
project(triton_atlas_backend LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20 CACHE STRING "C++ standard; must match Triton's")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# The installed Triton wheel: headers in include/, libtriton in _C/.
find_package(Python COMPONENTS Interpreter REQUIRED)
execute_process(
  COMMAND ${Python_EXECUTABLE} -c
          "import importlib.util; print(importlib.util.find_spec('triton').submodule_search_locations[0])"
  OUTPUT_VARIABLE TRITON_WHEEL_DIR OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE _triton_found)
set(TRITON_LIB "${TRITON_WHEEL_DIR}/_C/libtriton.so")
if(NOT _triton_found EQUAL 0 OR NOT EXISTS "${TRITON_WHEEL_DIR}/include/triton/Tools/PluginUtils.h")
  message(FATAL_ERROR "No Triton wheel with plugin headers found by ${Python_EXECUTABLE}")
endif()

# LLVM: headers, TableGen and CMake modules only. It must be the commit Triton was built with (scripts/build.sh checks).
if(NOT LLVM_INSTALL_DIR)
  set(LLVM_INSTALL_DIR "$ENV{LLVM_INSTALL_DIR}")
endif()
if(NOT LLVM_INSTALL_DIR)
  message(FATAL_ERROR "Set LLVM_INSTALL_DIR to an LLVM install; see scripts/build.sh")
endif()
find_package(MLIR REQUIRED CONFIG PATHS "${LLVM_INSTALL_DIR}/lib/cmake/mlir" NO_DEFAULT_PATH)

# The flag changes the mangled type of every ilist_iterator, and libtriton is built with it on. A mismatch only shows
# up as undefined symbols when Python loads the plugin.
file(STRINGS "${LLVM_INCLUDE_DIR}/llvm/Config/abi-breaking.h" _llvm_abi
     REGEX "^#define LLVM_ENABLE_ABI_BREAKING_CHECKS 1")
if(NOT _llvm_abi)
  message(FATAL_ERROR "${LLVM_INSTALL_DIR} needs LLVM_ENABLE_ABI_BREAKING_CHECKS=1 to match libtriton")
endif()

# atlas-mlir (pinned submodule) builds with its own CMake; EXCLUDE_FROM_ALL builds only the library we link.
add_subdirectory(third_party/atlas-mlir ${CMAKE_CURRENT_BINARY_DIR}/atlas-mlir EXCLUDE_FROM_ALL)

add_library(TritonAtlasBackend SHARED plugin.cpp)
add_dependencies(TritonAtlasBackend MLIRAtlas)
set_target_properties(TritonAtlasBackend PROPERTIES OUTPUT_NAME triton_atlas_backend)
target_include_directories(TritonAtlasBackend SYSTEM PRIVATE
  ${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS} "${TRITON_WHEEL_DIR}/include")
# atlas-mlir sets its include paths per folder, not on MLIRAtlas, so name them here (headers + TableGen output).
target_include_directories(TritonAtlasBackend PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/atlas-mlir/include
  ${CMAKE_CURRENT_BINARY_DIR}/atlas-mlir/include)
target_compile_options(TritonAtlasBackend PRIVATE
  -fvisibility=hidden -fno-rtti -fno-exceptions -Wno-deprecated-declarations)
# Name the two static libraries by file: linking the MLIRAtlas target would also pull in its static MLIR libraries.
# --no-undefined proves at link time that libtriton has everything; --no-as-needed keeps it listed as a dependency.
target_link_libraries(TritonAtlasBackend PRIVATE
  $<TARGET_FILE:MLIRAtlas>
  "${LLVM_LIBRARY_DIR}/libMLIRFuncInlinerExtension.a"
  -Wl,--no-undefined -Wl,--gc-sections
  -Wl,--no-as-needed "${TRITON_LIB}" -Wl,--as-needed)

install(TARGETS TritonAtlasBackend LIBRARY DESTINATION triton_atlas_backend COMPONENT plugin)
