# Every .cpp under this folder belongs to the plugin, so adding a file to a pass needs no edit here (one
# folder per pass; several people can add files without touching a shared list). CONFIGURE_DEPENDS re-runs the
# glob when files are added or removed.
file(GLOB_RECURSE TRITON_TO_ATLAS_VIRTUAL_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
target_sources(TritonAtlasBackend PRIVATE ${TRITON_TO_ATLAS_VIRTUAL_SOURCES})
