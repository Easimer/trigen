set(SRC
	stdafx.h
	entry.cpp

	application.cpp
	application.h
)

add_executable(mesh_paint_proto ${SRC})
target_include_directories(mesh_paint_proto PRIVATE ${SDL2_INCLUDE_DIRS} ${SDL2TTF_INCLUDE_DIR})
target_link_directories(mesh_paint_proto PRIVATE ${SDL2TTF_LIBRARY})
target_link_libraries(mesh_paint_proto PRIVATE
	${SDL2_LIBRARIES}
	SDL2_ttf
	glad
	imgui
	psp
	renderer
	arcball_camera
)
target_precompile_headers(mesh_paint_proto PRIVATE "stdafx.h")
tri_builddir(mesh_paint_proto)
