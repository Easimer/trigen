add_library(mesh_compress SHARED)

target_sources(mesh_compress
    PRIVATE
        src/mesh_compress.cpp
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
)

target_compile_definitions(mesh_compress
    PRIVATE
        TMC_COMPILATION=1)

set_target_properties(mesh_compress PROPERTIES FOLDER "Libs")
tri_builddir(mesh_compress)

