set(SRC_MC33
    src/mc33/libMC33++.cpp
    src/mc33/MC33.h
)

set(SRC
    include/marching_cubes.h
    src/main.cpp
)

add_library(marching_cubes STATIC ${SRC} ${SRC_MC33})
target_compile_definitions(marching_cubes PRIVATE -D_USE_MATH_DEFINES)
set_target_properties(marching_cubes PROPERTIES FOLDER "Libs")
target_include_directories(marching_cubes PUBLIC include/)
