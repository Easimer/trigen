add_executable(demo)
target_sources(demo PRIVATE
	iapplication.h
    demo.cpp

    scene.cpp
    scene.h
    scene_loader.cpp
    scene_loader.h
    playback.cpp
    playback.h
    async_image_loader.h
)

target_link_libraries(demo PRIVATE
	topo trigen mesh_compress
    arcball_camera stb CONAN_PKG::boost
    CONAN_PKG::nlohmann_json
    CONAN_PKG::libuv
)
tri_builddir(demo)