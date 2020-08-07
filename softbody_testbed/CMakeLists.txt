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

	events.h

	arcball_camera.cpp
	arcball_camera.h

	../public/raymarching.h
)

add_executable(softbody_testbed ${SRC})
target_include_directories(softbody_testbed PRIVATE ${SDL2_INCLUDE_DIRS} ${SDL2TTF_INCLUDE_DIR})
target_link_directories(softbody_testbed PRIVATE ${SDL2TTF_LIBRARY})
target_link_libraries(softbody_testbed PRIVATE ${SDL2_LIBRARIES} SDL2_ttf glad imgui tree_meshifier softbody)
target_compile_options(softbody_testbed PRIVATE "-mfma")
target_precompile_headers(softbody_testbed PRIVATE "stdafx.h")
tri_builddir(softbody_testbed)
