set(SRC
	main.cpp
)

add_executable(ifs_test ${SRC})
target_include_directories(ifs_test PRIVATE ${SDL2_INCLUDE_DIRS} ${SDL2TTF_INCLUDE_DIR})
target_link_directories(ifs_test PRIVATE ${SDL2TTF_LIBRARY})
target_link_libraries(ifs_test PRIVATE ${SDL2_LIBRARIES} SDL2_ttf)
tri_builddir(ifs_test)

