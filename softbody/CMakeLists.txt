set(SRC_PUBLIC
    ../public/softbody.h
)

set(SRC_IMPL
    s_simulation.cpp
)

add_library(softbody STATIC ${SRC_PUBLIC} ${SRC_IMPL})