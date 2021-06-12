option(RENDERER_ENABLE_TRACY "Enable the Tracy profiler for the renderer")

set(SRC_SHADER_SOURCES
    ${CMAKE_SOURCE_DIR}/shaders/generic.vsh.glsl
    ${CMAKE_SOURCE_DIR}/shaders/generic.fsh.glsl
    ${CMAKE_SOURCE_DIR}/shaders/lines.vsh.glsl
    ${CMAKE_SOURCE_DIR}/shaders/lines.fsh.glsl
    ${CMAKE_SOURCE_DIR}/shaders/points.vsh.glsl
    ${CMAKE_SOURCE_DIR}/shaders/points.fsh.glsl
    ${CMAKE_SOURCE_DIR}/shaders/deferred.vsh.glsl
    ${CMAKE_SOURCE_DIR}/shaders/deferred.fsh.glsl
)

embed_file(shaders.cpp ${SRC_SHADER_SOURCES})

add_library(renderer STATIC
    include/r_renderer.h
    include/r_queue.h
    src/r_gl.cpp
    src/r_gl_shadercompiler.cpp
    src/r_gl_shadercompiler.h
    src/r_sdl.cpp
    src/shader_program_builder.cpp
    src/shader_program_builder.h
    src/shader_generic.cpp
    src/shader_generic.h
    src/gl_utils.h
    src/gbuffer.cpp
    src/gbuffer.h
    src/gl.h
    src/glres.h
    src/sdl_helper.h
    shaders.cpp
)
target_link_libraries(renderer
    PUBLIC
        CONAN_PKG::glm
        SDL::SDL2
    PRIVATE
        glad imgui Tracy::Client virtfs
        mesh_compress
)

target_include_directories(renderer
    PUBLIC
        include/
)

if(${RENDERER_ENABLE_TRACY})
    target_compile_definitions(renderer PRIVATE TRACY_ENABLE=1)
endif()
set_target_properties(renderer PROPERTIES FOLDER "Libs")
target_precompile_headers(renderer PRIVATE src/stdafx.h)

add_custom_command(
    TARGET renderer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:SDL::SDL2>
        ${CMAKE_BINARY_DIR}
)
