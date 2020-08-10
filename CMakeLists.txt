cmake_minimum_required(VERSION 3.16)
project(trigen)

# =======================================================
# Purpose: Set the target's build directory
# =======================================================
macro(tri_builddir target)
	# For executables
	set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/Debug")
	set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/Release")
	set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_BINARY_DIR}/RelWithDebInfo")
	# For libraries
	set_target_properties(${target} PROPERTIES LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/Debug/bin")
	set_target_properties(${target} PROPERTIES LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/Release/bin")
	set_target_properties(${target} PROPERTIES LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_BINARY_DIR}/RelWithDebInfo/bin")
endmacro()

# =======================================================
# Purpose: Enable precompiled headers for target
# TODO(danielm): cmake has builtin support for this now!
# =======================================================
macro(tri_enable_pch target pchbase)
	if(MSVC)
		set_target_properties(${target} PROPERTIES COMPILE_FLAGS "/Yu\"${pchbase}.h\"")
		set_source_files_properties(${pchbase}.cpp
			PROPERTIES COMPILE_FLAGS "/Yc\"${pchbase}.h\""
			OBJECT_CREATES "${pchbase}.pch")
	endif()
endmacro()

# =======================================================
# Purpose: Makes a file not use the precompiled header
# =======================================================
macro(tri_ignore_pch file)
	if(MSVC)
		set_source_files_properties(${file}
			PROPERTIES COMPILE_FLAGS "/Y-")
	endif()
endmacro()

# =======================================================
# Purpose: Don't link the CRT
# =======================================================
macro(tri_no_crt target)
	if(MSVC)
		set_property(TARGET ${target} APPEND_STRING PROPERTY LINK_FLAGS_DEBUG "/NODEFAULTLIB:msvcrt ")
		set_property(TARGET ${target} APPEND_STRING PROPERTY LINK_FLAGS_RELEASE "/NODEFAULTLIB:msvcrt ")
		set_property(TARGET ${target} APPEND_STRING PROPERTY LINK_FLAGS_RELWITHDEBINFO "/NODEFAULTLIB:msvcrt ")
	endif()
endmacro()

# =======================================================
# Purpose: Defines a macro
# =======================================================
macro(tri_is_library target)
	target_compile_definitions(${target} PRIVATE ZAIDAN_LIBRARY=1)
endmacro()

# =======================================================
# Purpose: Sets the subsystem of the target
# =======================================================
macro(tri_subsystem target subsys)
	if(MSVC)
		set_property(TARGET ${target} APPEND_STRING PROPERTY LINK_FLAGS_DEBUG "/SUBSYSTEM:${subsys} ")
		set_property(TARGET ${target} APPEND_STRING PROPERTY LINK_FLAGS_RELEASE "/SUBSYSTEM:${subsys} ")
		set_property(TARGET ${target} APPEND_STRING PROPERTY LINK_FLAGS_RELWITHDEBINFO "/SUBSYSTEM:${subsys} ")
	endif(MSVC)
endmacro()

# =======================================================
# Purpose: Disables Runtime Type Information, exceptions
# and anything else that we don't need
# =======================================================
macro(tri_disable_bloat target)
	if(MSVC)
		set_property(TARGET ${target} APPEND_STRING PROPERTY COMPILE_OPTIONS "-GS-;-Gs9999999;-GR-;-Gm-;/EHs-c-;-fp:fast;-fp:except-")
		set_property(TARGET ${target} APPEND_STRING PROPERTY LINK_FLAGS_DEBUG "-opt:ref -STACK:0x400000,0x400000 ")
		set_property(TARGET ${target} APPEND_STRING PROPERTY LINK_FLAGS_RELEASE "-opt:ref -STACK:0x400000,0x400000 ")
		set_property(TARGET ${target} APPEND_STRING PROPERTY LINK_FLAGS_RELWITHDEBINFO "-opt:ref -STACK:0x400000,0x400000 ")
	endif(MSVC)
endmacro()

# =======================================================
# Purpose: Use this on every source file that uses
# SSE4.1 intrinsics
# =======================================================
macro(tri_sse41 file)
	if("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "x86")
		if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
				set_source_files_properties(${file} PROPERTIES COMPILE_FLAGS -msse4.1)
		endif()
	endif()
endmacro()

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Include public headers for all targets
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/public")

# Use folders
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

enable_testing()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

find_package(SDL2 REQUIRED)
find_package(SDL2_ttf REQUIRED)
find_package(OpenCL REQUIRED)

add_subdirectory(glad)
add_subdirectory(imgui)
add_subdirectory(renderer)
add_subdirectory(tree_meshifier)
add_subdirectory(softbody)
add_subdirectory(bark_test)
add_subdirectory(ifs_test)
add_subdirectory(cylindergen)
add_subdirectory(softbody_testbed)
add_subdirectory(editor)
add_subdirectory(testbed_qt)