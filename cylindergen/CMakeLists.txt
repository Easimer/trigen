set(SRC
	main.cpp
)

add_executable(cylindergen ${SRC})
target_include_directories(cylindergen PRIVATE ${SDL2_INCLUDE_DIRS} ${SDL2TTF_INCLUDE_DIR})
target_link_directories(cylindergen PRIVATE ${SDL2TTF_LIBRARY})
target_link_libraries(cylindergen PRIVATE ${SDL2_LIBRARIES} SDL2_ttf)
tri_builddir(cylindergen)
