set(SRC_GLAD
	glad.c
)

add_library(glad STATIC ${SRC_GLAD})
if(NOT WIN32)
    target_link_libraries(glad PUBLIC dl)
endif()
