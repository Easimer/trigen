option(SOFTBODY_ENABLE_CUDA "Enable CUDA backend")

if(${SOFTBODY_ENABLE_CUDA})
    # find_package(CUDA 11.0 REQUIRED)
    find_package(CUDAToolkit 11.0 REQUIRED)

    embed_file(cuda_templates.c ${CMAKE_CURRENT_SOURCE_DIR}/cuda_templates.cu)

    enable_language(CUDA)
    set(SRC_CUDA
        s_compute_cuda.cu
        s_compute_cuda_codegen.cpp
        cuda_templates.c
    )
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)

    set(SOFTBODY_CUDA_LIBS CUDA::cudart_static)
endif()

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

    l_iterators.h
    l_constexpr_map.h
    l_random.h
    s_iterators.cpp
    s_iterators.h

    c_particle.cpp
    c_particle.h

    s_compute_backend.h
    s_compute_ref.cpp
    s_compute_cl.cpp

    s_ext.h
    s_ext_plant.cpp
    s_ext_cloth.cpp

    f_serialization.cpp
    f_serialization.h

    ../public/raymarching.h
)

set(SRC_TESTS
    tests.main.cpp
    c_particle.tests.cpp
    f_serialization.tests.cpp
    m_utils.tests.cpp
)

add_library(softbody STATIC ${SRC_PUBLIC} ${SRC_IMPL} ${SRC_CUDA})
target_precompile_headers(softbody PRIVATE stdafx.h)
target_link_libraries(softbody PRIVATE ${OpenCL_LIBRARIES} ${SOFTBODY_CUDA_LIBS})
target_include_directories(softbody PRIVATE ${OpenCL_INCLUDE_DIRS})
add_compile_definitions(CL_HPP_TARGET_OPENCL_VERSION=120)
add_compile_definitions(CL_HPP_MINIMUM_OPENCL_VERSION=120)

add_executable(softbody_tests ${SRC_PUBLIC} ${SRC_IMPL} ${SRC_TESTS})
target_precompile_headers(softbody_tests PRIVATE stdafx.h)
tri_builddir(softbody_tests)
add_test(NAME softbody_tests COMMAND softbody_tests)
target_link_libraries(softbody_tests PRIVATE ${OpenCL_LIBRARIES})
target_include_directories(softbody_tests PRIVATE ${OpenCL_INCLUDE_DIRS})

if(${SOFTBODY_ENABLE_CUDA})
    target_compile_definitions(softbody PRIVATE SOFTBODY_CUDA_ENABLED=1 _GLIBCXX_USE_CXX11_ABI=0)
endif()
