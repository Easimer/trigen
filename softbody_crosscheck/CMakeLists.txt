set(SRC
    stdafx.h
    main.cpp

    cross_check.cpp
    cross_check.h

    benchmark.cpp
    benchmark.h
)

add_executable(sb_crosscheck ${SRC})
target_link_libraries(sb_crosscheck softbody)
target_precompile_headers(sb_crosscheck PRIVATE "stdafx.h")
