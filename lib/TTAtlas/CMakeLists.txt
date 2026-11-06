# Every .cpp under this folder belongs to the plugin (same rule as lib/TritonToAtlasVirtual/CMakeLists.txt).
file(GLOB_RECURSE TTATLAS_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
target_sources(TritonAtlasBackend PRIVATE ${TTATLAS_SOURCES})
