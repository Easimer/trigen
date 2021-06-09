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
		CONAN_PKG::glm
)

target_compile_definitions(mesh_compress
    PRIVATE
        TMC_COMPILATION=1)

set_target_properties(mesh_compress PROPERTIES FOLDER "Libs")
tri_builddir(mesh_compress)

