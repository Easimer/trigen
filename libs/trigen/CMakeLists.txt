set(SRC
    src/trigen.cpp
    include/trigen.h
    include/trigen.hpp
)

add_library(trigen SHARED ${SRC})
target_include_directories(trigen PUBLIC include/)
set_target_properties(trigen PROPERTIES FOLDER "Libs")
tri_builddir(trigen)