find_package(FBXSDK REQUIRED)

set(SRC
	stdafx.h
	entry.cpp

	application.cpp
	application.h
	session.cpp
	session.h
	fbx_export.cpp
	fbx_export.h
    fbx_standard_surface.cpp
    fbx_standard_surface.h
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
	Boost::thread
	FbxSdk::FbxSdk
)
target_precompile_headers(mesh_paint_proto PRIVATE "stdafx.h")
tri_builddir(mesh_paint_proto)
