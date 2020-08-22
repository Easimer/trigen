set(SRC_DC
    dual_contouring.cpp
    octree.h
    ../public/dual_contouring.h
)

add_library(dual_contouring STATIC ${SRC_DC})
