add_executable(demo)
target_sources(demo PRIVATE
    demo.cpp

    scene.cpp
    scene.h

    scene_basic_cube.cpp
)

target_link_libraries(demo PRIVATE topo trigen mesh_compress arcball_camera)
tri_builddir(demo)