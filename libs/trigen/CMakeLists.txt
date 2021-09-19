set(SRC
    src/trigen.cpp
    include/trigen.h
    include/trigen.hpp

    src/c_compile_test.c
)

add_library(trigen-static STATIC ${SRC})
target_include_directories(trigen-static PUBLIC include/)
target_link_libraries(trigen-static PRIVATE marching_cubes psp softbody mesh_compress foliage)
set_target_properties(trigen-static PROPERTIES FOLDER "Libs")

add_library(trigen SHARED src/empty.cpp include/trigen.h)
target_include_directories(trigen PUBLIC include/)
target_link_libraries(trigen PUBLIC trigen-static)
set_target_properties(trigen PROPERTIES FOLDER "Libs")
tri_builddir(trigen)
install(TARGETS trigen)

if(${TRIGEN_BUILD_CSHARP_BINDING})
set(SRC_DOTNET
    src/trigen.net.cpp
)
add_library(trigen-net SHARED ${SRC_DOTNET})
target_link_libraries(trigen-net PRIVATE trigen-static)
set_target_properties(trigen-net PROPERTIES
    FOLDER "Libs"
    COMMON_LANGUAGE_RUNTIME ""
)
tri_builddir(trigen-net)
install(TARGETS trigen-net)
endif()
