set(SRC
	stdafx.h
	entry.cpp

	application.cpp
	application.h

	softbody_renderer.cpp
	softbody_renderer.h

	r_renderer.h
	r_queue.h
	r_gl.cpp
)

set(SRC_SHADERS
	generic.vsh.glsl
	generic.fsh.glsl
)

add_executable(editor ${SRC})
target_include_directories(editor PRIVATE ${SDL2_INCLUDE_DIRS} ${SDL2TTF_INCLUDE_DIR})
target_link_directories(editor PRIVATE ${SDL2TTF_LIBRARY})
target_link_libraries(editor PRIVATE ${SDL2_LIBRARIES} SDL2_ttf glad imgui tree_meshifier softbody)
target_compile_options(editor PRIVATE "-mfma")
target_precompile_headers(editor PRIVATE "stdafx.h")
tri_builddir(editor)