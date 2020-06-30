set(SRC_PUBLIC
    ../public/softbody.h
)

set(SRC_IMPL
    stdafx.h
    common.h

    s_simulation.cpp
    s_simulation.h

    m_utils.cpp
    m_utils.h

    s_iterators.cpp
)

add_library(softbody STATIC ${SRC_PUBLIC} ${SRC_IMPL})
target_precompile_headers(softbody PRIVATE stdafx.h)