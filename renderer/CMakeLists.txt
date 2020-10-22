embed_file(shader_ellipsoid_vsh.cpp ${CMAKE_SOURCE_DIR}/shaders/ellipsoid.vsh.glsl)
embed_file(shader_ellipsoid_fsh.cpp ${CMAKE_SOURCE_DIR}/shaders/ellipsoid.fsh.glsl)
embed_file(shader_generic_vsh.cpp ${CMAKE_SOURCE_DIR}/shaders/generic.vsh.glsl)
embed_file(shader_generic_fsh.cpp ${CMAKE_SOURCE_DIR}/shaders/generic.fsh.glsl)
embed_file(shader_lines_vsh.cpp ${CMAKE_SOURCE_DIR}/shaders/lines.vsh.glsl)
embed_file(shader_lines_fsh.cpp ${CMAKE_SOURCE_DIR}/shaders/lines.fsh.glsl)
embed_file(shader_points_vsh.cpp ${CMAKE_SOURCE_DIR}/shaders/points.vsh.glsl)
embed_file(shader_points_fsh.cpp ${CMAKE_SOURCE_DIR}/shaders/points.fsh.glsl)

set(SRC_SHADERS
    shader_ellipsoid_vsh.cpp shader_ellipsoid_fsh.cpp
    shader_generic_vsh.cpp shader_generic_fsh.cpp
    shader_lines_vsh.cpp shader_lines_fsh.cpp
    shader_points_vsh.cpp shader_points_fsh.cpp
)

set(SRC
    ../public/r_renderer.h
    ../public/r_queue.h
    r_factory.cpp
    r_gl.cpp
    r_sdl.cpp
)

source_group("Embedded shaders" FILES ${SRC_SHADERS})

add_library(renderer STATIC ${SRC} ${SRC_SHADERS})
target_link_libraries(renderer glad ${SDL2_LIBRARIES} imgui)
target_include_directories(renderer PRIVATE ${SDL2_INCLUDE_DIRS})
