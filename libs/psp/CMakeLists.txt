set(SRC_PSP
    src/stdafx.h
    src/psp.cpp
    src/uv_unwrap.cpp
    src/uv_unwrap.h
    src/painting.cpp
    src/painting.h
    src/uv_quadtree.cpp
    src/uv_quadtree.h
)

option(PSP_ENABLE_TRACY "Enable Tracy in the psp library")

add_library(psp STATIC ${SRC_PSP})
target_link_libraries(psp PUBLIC CONAN_PKG::glm PRIVATE intersect Boost::boost Boost::thread Tracy::Client)
target_precompile_headers(psp PRIVATE src/stdafx.h)
target_include_directories(psp PUBLIC include/)
set_target_properties(psp PROPERTIES FOLDER "Libs")

if(${PSP_ENABLE_TRACY})
    target_compile_definitions(psp PRIVATE TRACY_ENABLE=1)
endif()
