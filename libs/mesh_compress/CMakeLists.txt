add_library(mesh_compress SHARED)

target_sources(mesh_compress
    PRIVATE
        include/trigen/mesh_compress.h
        src/common.h
        src/arena.cpp
        src/arena.h
        src/compression.cpp
        src/resource_management.cpp
        src/dbgmsg.cpp
        src/dbgmsg.h
)

target_include_directories(mesh_compress
    PUBLIC
        include/
)

target_precompile_headers(mesh_compress
    PRIVATE
        src/stdafx.h
)

target_link_libraries(mesh_compress
    PUBLIC
        CONAN_PKG::hedley
    PRIVATE
        CONAN_PKG::glm tracy_client
)

target_compile_definitions(mesh_compress
    PRIVATE
        TMC_COMPILATION=1)

set_target_properties(mesh_compress PROPERTIES FOLDER "Libs")
tri_builddir(mesh_compress)

set(MESH_COMPRESS_TRACY OFF CACHE BOOL "Build mesh_compress with Tracy instrumentation enabled")
if(${MESH_COMPRESS_TRACY})
target_compile_definitions(mesh_compress
    PRIVATE
        TRACY_ENABLE=1
)
endif()

set(MESH_COMPRESS_BUILD_EXAMPLES OFF CACHE BOOL "Build example code for mesh_compress")
if(${MESH_COMPRESS_BUILD_EXAMPLES})
add_executable(mesh_compress_quad examples/quad.c)
target_link_libraries(mesh_compress_quad
    PRIVATE
        mesh_compress
)
set_target_properties(mesh_compress_quad PROPERTIES FOLDER "Libs\\mesh_compress examples")
tri_builddir(mesh_compress_quad)
endif()

