set(SRC_GLAD
	src/glad.c
)

add_library(glad STATIC ${SRC_GLAD})
if(NOT WIN32)
    target_link_libraries(glad PUBLIC dl)
endif()
target_include_directories(glad PUBLIC include/)
set_target_properties(glad PROPERTIES FOLDER "Libs")

target_disable_clang_tidy(glad)
