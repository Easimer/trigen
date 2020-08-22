set(SRC_DC
    dual_contouring.cpp
    octree.h
    ../public/dual_contouring.h
)

add_library(dual_contouring STATIC ${SRC_DC})

set(SRC_DC_TESTS
    main.tests.cpp
    octree.tests.cpp
)

add_executable(dual_contouring_tests ${SRC_DC} ${SRC_DC_TESTS})
tri_builddir(dual_contouring_tests)
add_test(NAME dual_contouring_tests COMMAND dual_contouring_tests)
