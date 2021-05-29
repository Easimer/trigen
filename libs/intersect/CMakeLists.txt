set(SRC
    src/ray_triangle.cpp
    src/ray_aabb.cpp
    src/ray_sphere.cpp
    include/intersect.h
)

add_library(intersect STATIC ${SRC})
target_include_directories(intersect PUBLIC include/)
target_link_libraries(intersect PUBLIC CONAN_PKG::glm)
set_target_properties(intersect PROPERTIES FOLDER "Libs")
