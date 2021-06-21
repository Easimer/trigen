set(SRC
    src/trigen.cpp
    include/trigen.h
    include/trigen.hpp

    src/c_compile_test.c
)

add_library(trigen SHARED ${SRC})
target_include_directories(trigen PUBLIC include/)
target_link_libraries(trigen PRIVATE marching_cubes psp softbody mesh_compress foliage)
set_target_properties(trigen PROPERTIES FOLDER "Libs")
tri_builddir(trigen)

install(TARGETS trigen)
