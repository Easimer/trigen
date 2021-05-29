option(RENDERER_ENABLE_TRACY "Enable the Tracy profiler for the renderer")

set(SRC_SHADER_SOURCES
	${CMAKE_SOURCE_DIR}/shaders/ellipsoid.vsh.glsl
	${CMAKE_SOURCE_DIR}/shaders/ellipsoid.fsh.glsl
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

set(SRC
    ../public/r_renderer.h
    ../public/r_queue.h
    r_factory.cpp
    r_gl.cpp
    r_gl_shadercompiler.h
    r_sdl.cpp
	shaders.cpp
)

add_library(renderer STATIC ${SRC} ${SRC_SHADERS})
target_link_libraries(renderer
	PUBLIC
		glm
	PRIVATE
		glad SDL::SDL2 imgui ${TRACY_LIBRARIES}
)

target_include_directories(renderer PRIVATE ${SDL2_INCLUDE_DIRS} ${TRACY_INCLUDE_DIRS})
if(${RENDERER_ENABLE_TRACY})
    target_compile_definitions(renderer PRIVATE TRACY_ENABLE=1)
endif()
set_target_properties(renderer PROPERTIES FOLDER "Libs")
target_precompile_headers(renderer PRIVATE stdafx.h)
