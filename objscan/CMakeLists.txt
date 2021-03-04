set(SRC
    stdafx.h
    ../public/tiny_obj_loader.h
    objscan.cpp
    ../public/objscan.h

    objscan_math.cpp
    objscan_math.h

    mesh.h
)

add_library(objscan STATIC ${SRC})
target_link_libraries(objscan PUBLIC Threads::Threads intersect)
target_precompile_headers(objscan PRIVATE stdafx.h)
set_target_properties(objscan PROPERTIES FOLDER "Libs")

add_executable(objscan-test test.cpp)
target_link_libraries(objscan-test PRIVATE objscan)
