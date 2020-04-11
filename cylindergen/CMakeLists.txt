set(SRC
	main.cpp
)

set(SRC_SHADERS
	generic.vsh.glsl
	generic.fsh.glsl
)

add_executable(cylindergen ${SRC} "trunk_generator.cpp")
target_include_directories(cylindergen PRIVATE ${SDL2_INCLUDE_DIRS} ${SDL2TTF_INCLUDE_DIR})
target_link_directories(cylindergen PRIVATE ${SDL2TTF_LIBRARY})
target_link_libraries(cylindergen PRIVATE ${SDL2_LIBRARIES} SDL2_ttf glad imgui)
target_compile_options(cylindergen PRIVATE "-mfma")
target_precompile_headers(cylindergen PRIVATE "stdafx.h")
tri_builddir(cylindergen)