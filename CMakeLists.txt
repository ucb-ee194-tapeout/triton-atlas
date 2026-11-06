# Builds the Atlas backend's plugin: one shared library that the stock Triton wheel loads at runtime.
# Recipe from triton-lang/triton-ext@268252b cmake/triton-ext.cmake and cmake/llvm-abi-breaking.cmake (MIT):
# compile against the wheel's headers and an LLVM install of the same commit, and link only against libtriton, which
# exports MLIR and LLVM. A second copy of MLIR in our library would break MLIR's type registry.
cmake_minimum_required(VERSION 3.22)

# LLVM: headers, TableGen and CMake modules only. It must be the commit Triton was built with (checked below).
# env.sh's value always wins. It's cached so a later `cmake --build` can reconfigure without env.sh.
if(DEFINED ENV{LLVM_INSTALL_DIR})
    set(LLVM_INSTALL_DIR "$ENV{LLVM_INSTALL_DIR}" CACHE PATH "LLVM install to build against" FORCE)
endif()
if(NOT EXISTS "${LLVM_INSTALL_DIR}/lib/cmake/mlir")
    message(
        FATAL_ERROR
        "No MLIR in '${LLVM_INSTALL_DIR}'. Fix LLVM_INSTALL_DIR in env.sh or run make setup.llvm LLVM=<dir>."
    )
endif()

# Compiler: the clang from that LLVM, as Triton uses, or the ones env.sh names. make and uv both get here, and uv skips
# scripts/common.sh, so this is the place to stop a shell's settings, such as conda's, from reaching our build.
foreach(
    _var
    CC
    CXX
    CFLAGS
    CXXFLAGS
    CPPFLAGS
    LDFLAGS
)
    unset(ENV{${_var}})
endforeach()
set(_cc "$ENV{TRITON_ATLAS_CC}")
set(_cxx "$ENV{TRITON_ATLAS_CXX}")
if(NOT _cxx AND EXISTS "${LLVM_INSTALL_DIR}/bin/clang++")
    set(_cc "${LLVM_INSTALL_DIR}/bin/clang")
    set(_cxx "${LLVM_INSTALL_DIR}/bin/clang++")
endif()
if(NOT CMAKE_CXX_COMPILER AND _cxx)
    set(CMAKE_C_COMPILER "${_cc}" CACHE FILEPATH "C compiler")
    set(CMAKE_CXX_COMPILER "${_cxx}" CACHE FILEPATH "C++ compiler")
endif()

project(triton_atlas_backend LANGUAGES C CXX)

# Link with that LLVM's lld, if it has one, when compiling with its clang.
if(CMAKE_CXX_COMPILER STREQUAL "${LLVM_INSTALL_DIR}/bin/clang++" AND EXISTS "${LLVM_INSTALL_DIR}/bin/ld.lld")
    add_link_options(-fuse-ld=lld)
endif()

set(CMAKE_CXX_STANDARD 20 CACHE STRING "C++ standard; must match Triton's")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# The installed Triton wheel: headers in include/, libtriton in _C/.
find_package(Python COMPONENTS Interpreter REQUIRED)
execute_process(
    COMMAND
        ${Python_EXECUTABLE} -c
        "import importlib.util; print(importlib.util.find_spec('triton').submodule_search_locations[0])"
    OUTPUT_VARIABLE TRITON_WHEEL_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _triton_found
)
set(TRITON_LIB "${TRITON_WHEEL_DIR}/_C/libtriton.so")
if(NOT _triton_found EQUAL 0 OR NOT EXISTS "${TRITON_WHEEL_DIR}/include/triton/Tools/PluginUtils.h")
    message(FATAL_ERROR "No Triton wheel with plugin headers found by ${Python_EXECUTABLE}")
endif()

# Forget a previous LLVM, so changing LLVM_INSTALL_DIR in an existing build folder takes effect.
unset(MLIR_DIR CACHE)
unset(LLVM_DIR CACHE)
find_package(MLIR REQUIRED CONFIG PATHS "${LLVM_INSTALL_DIR}/lib/cmake/mlir" NO_DEFAULT_PATH)

# Take Triton's requirements from the Triton we link: libtriton carries LLVM's own version line, and Version.h names
# the release. Nothing here needs updating when Triton moves.
file(STRINGS "${TRITON_LIB}" _triton_llvm REGEX "LLVM version [0-9.a-z]+ \\([0-9a-f]+\\)" LIMIT_COUNT 1)
string(REGEX MATCH "\\(([0-9a-f]+)\\)" _ "${_triton_llvm}")
set(TRITON_LLVM_COMMIT "${CMAKE_MATCH_1}")
file(STRINGS "${LLVM_INCLUDE_DIR}/llvm/Support/VCSRevision.h" _llvm_revision REGEX "define LLVM_REVISION")
string(REGEX MATCH "\"\\(?([0-9a-f]+)\\)?\"" _ "${_llvm_revision}") # plain or raw string, depending on the LLVM build
set(HAVE_LLVM_COMMIT "${CMAKE_MATCH_1}")
if(NOT TRITON_LLVM_COMMIT OR NOT TRITON_LLVM_COMMIT STREQUAL HAVE_LLVM_COMMIT)
    message(
        FATAL_ERROR
        "${LLVM_INSTALL_DIR} is LLVM '${HAVE_LLVM_COMMIT}', but ${TRITON_LIB} was built with "
        "'${TRITON_LLVM_COMMIT}'. Set LLVM_INSTALL_DIR in env.sh to an install of that commit, or to build/llvm and run make setup.llvm."
    )
endif()

# If the optional Triton checkout is here, it must be the release we link. The source says so itself; a shallow
# checkout has no tags to ask.
set(_triton_init "${CMAKE_CURRENT_SOURCE_DIR}/third_party/triton/python/triton/__init__.py")
if(EXISTS "${_triton_init}")
    file(STRINGS "${TRITON_WHEEL_DIR}/include/triton/Version.h" _triton_version REGEX "define TRITON_VERSION \"")
    string(REGEX MATCH "\"([^\"+]+)" _ "${_triton_version}")
    set(TRITON_RELEASE "${CMAKE_MATCH_1}")
    file(STRINGS "${_triton_init}" _triton_source REGEX "^__version__ = ")
    string(REGEX MATCH "'([^'+]+)" _ "${_triton_source}")
    set(TRITON_SOURCE_RELEASE "${CMAKE_MATCH_1}")
    if(NOT TRITON_SOURCE_RELEASE STREQUAL TRITON_RELEASE)
        message(
            FATAL_ERROR
            "third_party/triton is ${TRITON_SOURCE_RELEASE}, but the installed Triton is ${TRITON_RELEASE}"
        )
    endif()
endif()

# The flag changes the mangled type of every ilist_iterator, and libtriton is built with it on. A mismatch only shows
# up as undefined symbols when Python loads the plugin.
file(
    STRINGS
    "${LLVM_INCLUDE_DIR}/llvm/Config/abi-breaking.h"
    _llvm_abi
    REGEX "^#define LLVM_ENABLE_ABI_BREAKING_CHECKS 1"
)
if(NOT _llvm_abi)
    message(FATAL_ERROR "${LLVM_INSTALL_DIR} needs LLVM_ENABLE_ABI_BREAKING_CHECKS=1 to match libtriton")
endif()

list(APPEND CMAKE_MODULE_PATH ${MLIR_CMAKE_DIR} ${LLVM_CMAKE_DIR})
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include_directories(${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})

# Headers name their own paths, in LLVM's asserts for example, so a wheel would carry this machine's folders.
add_compile_options(
    "-ffile-prefix-map=${LLVM_INSTALL_DIR}=llvm"
    "-ffile-prefix-map=${TRITON_WHEEL_DIR}=triton"
    "-ffile-prefix-map=${CMAKE_SOURCE_DIR}=."
)

# atlas-mlir (pinned submodule) builds with its own CMake; EXCLUDE_FROM_ALL builds only the library we link.
add_subdirectory(third_party/atlas-mlir ${CMAKE_CURRENT_BINARY_DIR}/atlas-mlir EXCLUDE_FROM_ALL)

# Our own passes: declarations in include/, TableGen output in the build folder.
add_subdirectory(include)

add_library(TritonAtlasBackend SHARED plugin.cpp)
add_dependencies(TritonAtlasBackend MLIRAtlas TTAtlasIncGen TritonToAtlasVirtualPassIncGen)
set_target_properties(TritonAtlasBackend PROPERTIES OUTPUT_NAME triton_atlas_backend)
add_subdirectory(lib)
add_subdirectory(tools)
target_include_directories(
    TritonAtlasBackend
    SYSTEM
    PRIVATE ${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS} "${TRITON_WHEEL_DIR}/include"
)
# atlas-mlir sets its include paths per folder, not on MLIRAtlas, so name them here (headers + TableGen output).
target_include_directories(
    TritonAtlasBackend
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_BINARY_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/atlas-mlir/include
        ${CMAKE_CURRENT_BINARY_DIR}/atlas-mlir/include
)
target_compile_options(
    TritonAtlasBackend
    PRIVATE -fvisibility=hidden -fno-rtti -fno-exceptions -Wno-deprecated-declarations
)
# Name the two static libraries by file: linking the MLIRAtlas target would also pull in its static MLIR libraries.
# --no-undefined proves at link time that libtriton has everything; --no-as-needed keeps it listed as a dependency.
target_link_libraries(
    TritonAtlasBackend
    PRIVATE
        $<TARGET_FILE:MLIRAtlas>
        "${LLVM_LIBRARY_DIR}/libMLIRFuncInlinerExtension.a"
        -Wl,--no-undefined
        -Wl,--gc-sections
        -Wl,--no-as-needed
        "${TRITON_LIB}"
        -Wl,--as-needed
)

# clangd finds compile_commands.json at the repo root with no settings. Triton's setup.py links it the same way.
file(CREATE_LINK "${CMAKE_BINARY_DIR}/compile_commands.json" "${CMAKE_SOURCE_DIR}/compile_commands.json" SYMBOLIC)
# The VS Code MLIR extension looks in build/ for this one, and its database setting doesn't expand ${workspaceFolder}.
file(
    CREATE_LINK
        "${CMAKE_BINARY_DIR}/tablegen_compile_commands.yml"
        "${CMAKE_BINARY_DIR}/../tablegen_compile_commands.yml"
    SYMBOLIC
)

install(TARGETS TritonAtlasBackend LIBRARY DESTINATION triton_atlas_backend COMPONENT plugin)
