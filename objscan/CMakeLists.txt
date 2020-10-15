set(SRC
    tiny_obj_loader.h
    objscan.cpp
    ../public/objscan.h
)

add_library(objscan STATIC ${SRC})

add_executable(objscan-test test.cpp)
target_link_libraries(objscan-test PRIVATE objscan)
