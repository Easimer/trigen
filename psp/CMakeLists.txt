set(SRC_PSP
    src/psp.cpp
    src/uv_unwrap.cpp
    src/uv_unwrap.h
)

add_library(psp STATIC ${SRC_PSP})
target_include_directories(psp PUBLIC include/)
set_target_properties(psp PROPERTIES FOLDER "Libs")