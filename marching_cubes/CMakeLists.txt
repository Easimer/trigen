set(SRC_MC33
    mc33/libMC33++.cpp
    mc33/MC33.h
)

set(SRC
    ../public/marching_cubes.h
    main.cpp
)

add_library(marching_cubes STATIC ${SRC} ${SRC_MC33})
target_compile_definitions(marching_cubes PRIVATE -D_USE_MATH_DEFINES)
set_target_properties(marching_cubes PROPERTIES FOLDER "Libs")
