set(SRC
	stdafx.h
	entry.cpp

	application.cpp
	application.h
	session.cpp
	session.h
)

embed_file(test_grid.cpp ${CMAKE_CURRENT_SOURCE_DIR}/test_grid.png)

set(SRC_RESOURCES
	test_grid.cpp
)

add_executable(mesh_paint_proto ${SRC} ${SRC_RESOURCES})
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
	softbody
	softbody_ext
	marching_cubes
	stb
	mesh_export
	trigen
	CONAN_PKG::boost
	RCmd::Base
	RCmd::Softbody
)
target_precompile_headers(mesh_paint_proto PRIVATE "stdafx.h")
tri_builddir(mesh_paint_proto)

install(TARGETS mesh_paint_proto)

add_custom_command(
    TARGET mesh_paint_proto POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:SDL::SDL2>
        ${CMAKE_BINARY_DIR}
)
