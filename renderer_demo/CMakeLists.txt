add_executable(renderer_demo)

target_sources(renderer_demo
    PRIVATE
        ds_entry.cpp
        ds_world.cpp ds_world.hpp
)

target_link_libraries(renderer_demo
    PRIVATE
		topo
        RCmd::Base
        stb
        arcball_camera
)

tri_builddir(renderer_demo)