set(SRC
    ../public/r_renderer.h
    ../public/r_queue.h
    r_factory.cpp
    r_gl.cpp
    r_sdl.cpp
)

add_library(renderer STATIC ${SRC})
target_link_libraries(renderer glad ${SDL2_LIBRARIES})
target_include_directories(renderer PRIVATE ${SDL2_INCLUDE_DIRS})
