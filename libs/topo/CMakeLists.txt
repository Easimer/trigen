add_library(topo SHARED)

target_include_directories(topo PUBLIC include/)

target_sources(topo
    PRIVATE
        src/stdafx.h

        src/topo_gl.cpp
        src/topo_sdl.cpp
        src/render_queue.cpp
        src/render_queue.h
        src/gl_model_manager.cpp
        src/gl_model_manager.h
        src/gl_texture_manager.cpp
        src/gl_texture_manager.h
        src/gl_shadercompiler.cpp
        src/gl_shadercompiler.h
        src/gl_depth_prepass.cpp
        src/gl_depth_prepass.h
        src/material_manager.cpp
        src/material_manager.h
        src/renderable_manager.cpp
        src/renderable_manager.h
        src/shader_generic.cpp
        src/shader_generic.h
        src/shader_program_builder.cpp
        src/shader_program_builder.h
        src/gl_gbuffer.cpp
        src/gl_gbuffer.h
        src/gl_color_pass.cpp
        src/gl_color_pass.h
        src/gl_multidraw.cpp
        src/gl_multidraw.h

        include/topo.h
        include/topo_sdl.h
)

target_compile_definitions(topo
    PRIVATE
        TOPO_BUILDING=1
        GLRES_GLM=1
        TRACY_ENABLE=1
)

target_link_libraries(topo
    PUBLIC
        CONAN_PKG::glm
        SDL::SDL2
        imgui
    PRIVATE
        glad mesh_compress virtfs tracy_client
)

tri_builddir(topo)

set(SRC_SHADER_SOURCES
    ${CMAKE_SOURCE_DIR}/shaders/generic.vsh.glsl
    ${CMAKE_SOURCE_DIR}/shaders/generic.fsh.glsl
    ${CMAKE_SOURCE_DIR}/shaders/points.vsh.glsl
    ${CMAKE_SOURCE_DIR}/shaders/points.fsh.glsl
    ${CMAKE_SOURCE_DIR}/shaders/transparent.vsh.glsl
    ${CMAKE_SOURCE_DIR}/shaders/transparent.fsh.glsl
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/depth_pass.vsh.glsl
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/depth_pass.fsh.glsl
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/solid_color.vsh.glsl
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/solid_color.fsh.glsl
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/lines2.vsh.glsl
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/lines2.fsh.glsl
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/textured_unlit.vsh.glsl
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/textured_unlit.fsh.glsl
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/textured_lit.vsh.glsl
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/textured_lit.fsh.glsl
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/gbuffer_merge.vsh.glsl
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/gbuffer_merge.fsh.glsl
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/model_matrix_compute.glsl
)

embed_file(shaders.cpp ${SRC_SHADER_SOURCES})
target_sources(topo PRIVATE shaders.cpp)

target_precompile_headers(topo PRIVATE src/stdafx.h)
