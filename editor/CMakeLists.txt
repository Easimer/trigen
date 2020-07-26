set(SRC
	stdafx.h
	entry.cpp

	application.cpp
	application.h
)

add_executable(editor ${SRC})
target_include_directories(editor PRIVATE ${SDL2_INCLUDE_DIRS} ${SDL2TTF_INCLUDE_DIR})
target_link_directories(editor PRIVATE ${SDL2TTF_LIBRARY})
target_link_libraries(editor PRIVATE ${SDL2_LIBRARIES} SDL2_ttf glad imgui tree_meshifier softbody)
target_compile_options(editor PRIVATE "-mfma")
target_precompile_headers(editor PRIVATE "stdafx.h")
tri_builddir(editor)
