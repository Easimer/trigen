set(SRC
    src/trigen.cpp
    include/trigen.h
    include/trigen.hpp

    src/c_compile_test.c
)

set(SRC_CSHARP
    src/trigen.cs
    src/Properties/AssemblyInfo.cs
)

add_library(trigen SHARED ${SRC})
target_include_directories(trigen PUBLIC include/)
target_link_libraries(trigen PRIVATE marching_cubes psp softbody)
set_target_properties(trigen PROPERTIES FOLDER "Libs")
tri_builddir(trigen)

install(TARGETS trigen)

if(${TRIGEN_BUILD_CSHARP_BINDING})
    add_library(trigen-net SHARED ${SRC_CSHARP})
	set_target_properties(trigen-net PROPERTIES FOLDER "Libs")
	install(TARGETS trigen-net)
endif()

