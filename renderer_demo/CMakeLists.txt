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

add_custom_target(renderer_assets
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_LIST_DIR}/assets ${CMAKE_BINARY_DIR}/assets
)
add_dependencies(renderer_demo renderer_assets)