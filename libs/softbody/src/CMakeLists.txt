# Embed the OpenCL source in the executable
embed_file(shape_matching.cl.cpp ${CMAKE_SOURCE_DIR}/shaders/shape_matching.cl)

add_library(softbody STATIC
    stdafx.h
    types.h
    system_state.h
    system_state.generated.h
    logger.h
    collider_handles.h
    collision_constraint.h

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
    s_ext_rope.cpp

    f_serialization.cpp
    f_serialization.h

    ../include/softbody.h

    ../../../public/raymarching.h

    shape_matching.cl.cpp
)

target_precompile_headers(softbody PRIVATE stdafx.h)
set_target_properties(softbody PROPERTIES FOLDER "Libs")

target_compile_definitions(softbody PRIVATE CL_HPP_TARGET_OPENCL_VERSION=120 CL_HPP_MINIMUM_OPENCL_VERSION=120)
if(${SOFTBODY_ENABLE_TRACY})
    target_compile_definitions(softbody PRIVATE TRACY_ENABLE=1)
endif()

target_link_libraries(softbody
    PUBLIC
        CONAN_PKG::glm
    PRIVATE
        ${OpenCL_LIBRARIES} ${TRACY_LIBRARIES} intersect
        metadesk
)
target_include_directories(softbody PRIVATE ${OpenCL_INCLUDE_DIRS})

if(${BUILD_TESTING})
	add_executable(softbody_tests
		tests.main.cpp
        f_serialization.tests.cpp
	)
	target_precompile_headers(softbody_tests PRIVATE stdafx.h)
	tri_builddir(softbody_tests)
	add_test(NAME softbody_tests COMMAND softbody_tests)
	target_link_libraries(softbody_tests PRIVATE softbody)
endif()

option(SOFTBODY_BUILD_DEMO_EXTENSIONS OFF)

if(${SOFTBODY_BUILD_DEMO_EXTENSIONS})
    target_compile_definitions(softbody PRIVATE SOFTBODY_BUILD_DEMO_EXTENSIONS=1)
endif()

if(${SOFTBODY_ENABLE_CUDA})
    target_compile_definitions(softbody PRIVATE SOFTBODY_CUDA_ENABLED=1 _GLIBCXX_USE_CXX11_ABI=0)
    find_package(CUDAToolkit 11.0 REQUIRED)

    add_subdirectory(rt_intersect)

    embed_file(cuda_templates.cpp ${CMAKE_CURRENT_SOURCE_DIR}/cuda_templates.cu)

    enable_language(CUDA)

    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)

    target_sources(softbody PRIVATE
        s_compute_cuda.cpp
        s_compute_cuda.h
        s_compute_cuda_kernels.cu
        s_compute_cuda_codegen.cpp
        cuda_templates.cpp
        cuda_memtrack.cpp cuda_memtrack.h
    )

    target_link_libraries(softbody PRIVATE
		CUDA::cudart_static CUDA::nvrtc CUDA::cuda_driver CUDA::nvToolsExt rt_intersect
    )

    if(${SOFTBODY_CUDA_MEMTRACK})
        target_compile_definitions(softbody PRIVATE SOFTBODY_CUDA_MEMTRACK=1)
    endif()
endif()

add_executable(system_state_generator system_state_generator.cpp)
target_link_libraries(system_state_generator PRIVATE metadesk)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/system_state.generated.h
    COMMAND system_state_generator system_state.generated.h ${CMAKE_CURRENT_SOURCE_DIR}/system_state.md
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/system_state.md)
target_include_directories(softbody PRIVATE ${CMAKE_CURRENT_BINARY_DIR})