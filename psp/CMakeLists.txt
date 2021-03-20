set(SRC_PSP
    src/stdafx.h
    src/psp.cpp
    src/uv_unwrap.cpp
    src/uv_unwrap.h
    src/painting.cpp
    src/painting.h
)

add_library(psp STATIC ${SRC_PSP})
target_link_libraries(psp intersect Boost::boost)
target_precompile_headers(psp PRIVATE src/stdafx.h)
target_include_directories(psp PUBLIC include/)
set_target_properties(psp PROPERTIES FOLDER "Libs")
