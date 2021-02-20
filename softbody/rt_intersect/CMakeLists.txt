find_package(CUDA REQUIRED)
find_package(OptiX REQUIRED)

cuda_compile_ptx(
    PTX_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/programs.cu
    OPTIONS "-I\"${OPTIX_INCLUDE_DIR}\" --use_fast_math"
)

embed_file(programs.cpp cuda_compile_ptx_1_generated_programs.cu.ptx)

set(SRC
    src/optix_utils.h
    src/params.h
    include/rt_intersect.h
    src/rt_intersect.cpp

    programs.cpp
)

set(CUDA_LIBS CUDA::cudart_static CUDA::nvrtc CUDA::cuda_driver CUDA::nvToolsExt)

add_library(rt_intersect STATIC ${SRC})
target_include_directories(rt_intersect PRIVATE ${OPTIX_INCLUDE_DIR} PUBLIC include/)
target_link_libraries(rt_intersect PRIVATE ${CUDA_LIBS})
set_target_properties(rt_intersect PROPERTIES FOLDER "Libs")