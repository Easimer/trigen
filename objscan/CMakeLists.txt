embed_file(objscan.cl.cpp ${CMAKE_CURRENT_SOURCE_DIR}/objscan.cl)
set(SRC
    stdafx.h
    tiny_obj_loader.h
    objscan.cpp
    ../public/objscan.h

	objscan_math.cpp
	objscan_math.h

    objscan.cl.cpp
)

add_library(objscan STATIC ${SRC})
target_precompile_headers(objscan PRIVATE stdafx.h)

add_executable(objscan-test test.cpp)
target_link_libraries(objscan-test PRIVATE objscan)
