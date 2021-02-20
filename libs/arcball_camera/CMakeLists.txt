set(SRC
    src/arcball_camera.cpp
    include/arcball_camera.h
)

add_library(arcball_camera STATIC ${SRC})
target_include_directories(arcball_camera PUBLIC include/)
set_target_properties(arcball_camera PROPERTIES FOLDER "Libs")
