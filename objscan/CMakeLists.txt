set(SRC
    stdafx.h
    tiny_obj_loader.h
    objscan.cpp
    ../public/objscan.h

    objscan_math.cpp
    objscan_math.h

    intersection.cpp
    intersection.h
    mesh.h
)

add_library(objscan STATIC ${SRC})
target_link_libraries(objscan PUBLIC Threads::Threads)
target_precompile_headers(objscan PRIVATE stdafx.h)

add_executable(objscan-test test.cpp)
target_link_libraries(objscan-test PRIVATE objscan)
