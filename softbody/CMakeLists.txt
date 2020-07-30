set(SRC_PUBLIC
    ../public/softbody.h
)

set(SRC_IMPL
    stdafx.h
    common.h

    s_simulation.cpp
    s_simulation.h

    m_utils.cpp
    m_utils.h

    s_iterators.cpp

    c_particle.cpp
    c_particle.h

    s_compute_backend.h
    s_compute_ref.cpp
)

set(SRC_TESTS
    tests.main.cpp
    c_particle.tests.cpp
    m_utils.tests.cpp
)

add_library(softbody STATIC ${SRC_PUBLIC} ${SRC_IMPL})
target_precompile_headers(softbody PRIVATE stdafx.h)
add_sycl_to_target(softbody)
target_include_directories(softbody PRIVATE ${ComputeCpp_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIRS})

add_executable(softbody_tests ${SRC_PUBLIC} ${SRC_IMPL} ${SRC_TESTS})
target_precompile_headers(softbody_tests PRIVATE stdafx.h)
tri_builddir(softbody_tests)
add_test(NAME softbody_tests COMMAND softbody_tests)
add_sycl_to_target(softbody_tests)
target_include_directories(softbody_tests PRIVATE ${ComputeCpp_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIRS})
