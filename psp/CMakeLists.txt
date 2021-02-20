set(SRC_PSP
    src/psp.cpp
)

add_library(psp STATIC ${SRC_PSP})
target_include_directories(psp PUBLIC include/)
set_target_properties(psp PROPERTIES FOLDER "Libs")