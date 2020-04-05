set(SRC
	main.cpp
)

add_executable(bark_test ${SRC})
target_include_directories(bark_test PRIVATE ${SDL2_INCLUDE_DIRS})
target_link_libraries(bark_test PRIVATE ${SDL2_LIBRARIES})
tri_builddir(bark_test)