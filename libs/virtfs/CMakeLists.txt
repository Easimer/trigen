add_library(virtfs STATIC
    src/virtfs.cpp
    include/virtfs.hpp
)

target_include_directories(virtfs PUBLIC include/)
set_target_properties(virtfs PROPERTIES FOLDER "Libs")
