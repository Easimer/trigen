if(${TRIGEN_MODULAR_BUILD})
    add_library(foliage SHARED)
    target_compile_definitions(foliage PUBLIC TRIGEN_MODULAR_EXPORT)
else()
    add_library(foliage STATIC)
endif()

target_compile_definitions(foliage
    PRIVATE
        FOLIAGE_BUILDING
)

target_include_directories(foliage
    PUBLIC
        include
)
target_sources(foliage
    PRIVATE
        src/foliage.cpp include/foliage.hpp
)
target_link_libraries(foliage
	PUBLIC
        softbody CONAN_PKG::hedley mesh_compress
)
set_target_properties(foliage
    PROPERTIES
        FOLDER "Libs"
)
tri_builddir(foliage)