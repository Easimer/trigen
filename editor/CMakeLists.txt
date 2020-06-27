set(SRC
	stdafx.h
	entry.cpp

	render_queue.h

	main_window.cpp
	main_window.h
	framebuffer.cpp
	framebuffer.h
	tree_renderer.cpp
	tree_renderer.h
)

set(SRC_SHADERS
	generic.vsh.glsl
	generic.fsh.glsl
)

add_executable(editor ${SRC})
target_include_directories(editor PRIVATE ${SDL2_INCLUDE_DIRS} ${SDL2TTF_INCLUDE_DIR})
target_link_directories(editor PRIVATE ${SDL2TTF_LIBRARY})
target_link_libraries(editor PRIVATE ${SDL2_LIBRARIES} SDL2_ttf glad imgui tree_meshifier)
target_compile_options(editor PRIVATE "-mfma")
target_precompile_headers(editor PRIVATE "stdafx.h")
tri_builddir(editor)