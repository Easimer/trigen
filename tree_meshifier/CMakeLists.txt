set(SRC_TM
	tree_meshifier.cpp
	trunk_generator.cpp
	trunk_generator.h
	../public/trigen/tree_meshifier.h
)

add_library(tree_meshifier STATIC ${SRC_TM})
target_compile_options(tree_meshifier PRIVATE "-mfma")
target_compile_definitions(tree_meshifier PRIVATE -D_USE_MATH_DEFINES)
