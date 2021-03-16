# === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
#
# Purpose: CMake module for the Autodesk FBX SDK
#

# TODO: dynamic linking fbxsdk is not supported

if(NOT FBX_SDK_DIR)
	set(FBX_SDK_DIR "" CACHE PATH "FBX SDK directory (the one that contains 'include', 'lib' and 'samples'")
	message("Please set FBX_SDK_DIR")
else()
	if(NOT FBX_SDK_BUILD_TYPE)
		# Default to using the release library, even in debug mode, but let users
		# override FBX_SDK_BUILD_TYPE if they want to
		set(FBX_SDK_BUILD_TYPE "release" CACHE STRING "FBX SDK build type")
	endif()

	if(WIN32)
		set(FBX_SDK_PLAT "vs2017")
		set(FBX_SDK_LIBRARY_EXT ".lib")
		set(FBX_SDK_DLL_EXT ".dll")

		if(MSVC_RUNTIME_LIBRARY EQUAL "MultiThreaded")
			set(FBX_SDK_LIBRARY_SUFFIX "-mt")
		elseif(MSVC_RUNTIME_LIBRARY EQUAL "MultiThreadedDLL")
			set(FBX_SDK_LIBRARY_SUFFIX "-md")
		elseif(MSVC_RUNTIME_LIBRARY EQUAL "MultiThreadedDebug")
			set(FBX_SDK_LIBRARY_SUFFIX "-mt")
		elseif(MSVC_RUNTIME_LIBRARY EQUAL "MultiThreadedDebugDLL")
			set(FBX_SDK_LIBRARY_SUFFIX "-md")
		endif()
	endif(WIN32)

	if(UNIX AND NOT APPLE)
		set(FBX_SDK_PLAT "gcc")
		set(FBX_SDK_LIBRARY_EXT ".a")
		set(FBX_SDK_DLL_EXT ".so")
		# The Linux SDK doesn't come with libxml and zlib
		find_package(ZLIB REQUIRED)
		find_package(LibXml2 REQUIRED)
		list(APPEND FBX_SDK_LIBRARY_DEPS ${ZLIB_LIBRARIES} ${LIBXML2_LIBRARIES})
		list(APPEND FBX_SDK_LIBRARY ${ZLIB_LIBRARIES} ${LIBXML2_LIBRARIES})
		list(APPEND FBX_SDK_INCLUDE_DIR ${LIBXML2_INCLUDE_DIRS} ${ZLIB_INCLUDE_DIRS})
	endif(UNIX AND NOT APPLE)

	if(CMAKE_SIZEOF_VOID_P EQUAL 8)
		set(FBX_SDK_ARCH "x64")
	elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
		set(FBX_SDK_ARCH "x86")
	endif()

	set(FBX_SDK_LIB_ROOT "${FBX_SDK_DIR}/lib/${FBX_SDK_PLAT}/${FBX_SDK_ARCH}/${FBX_SDK_BUILD_TYPE}")
	get_filename_component(FBX_SDK_VERSION_STRING ${FBX_SDK_DIR} NAME)

	set(FBX_SDK_LIBRARY_FILE "${FBX_SDK_LIB_ROOT}/libfbxsdk${FBX_SDK_LIBRARY_SUFFIX}${FBX_SDK_LIBRARY_EXT}")
	list(APPEND FBX_SDK_LIBRARY ${FBX_SDK_LIBRARY_FILE})
	list(APPEND FBX_SDK_INCLUDE_DIR "${FBX_SDK_DIR}/include/")
	message("FBX SDK version: '${FBX_SDK_VERSION_STRING}' lib: ${FBX_SDK_LIBRARY} include: ${FBX_SDK_INCLUDE_DIR}")
	set(FBX_SDK_DLL "${FBX_SDK_LIB_ROOT}/libfbxsdk${FBX_SDK_LIBRARY_SUFFIX}${FBX_SDK_DLL_EXT}")

	if(NOT TARGET FbxSdk::FbxSdk)
		add_library(FbxSdk::FbxSdk UNKNOWN IMPORTED)
		set_target_properties(FbxSdk::FbxSdk PROPERTIES
			IMPORTED_LOCATION             ${FBX_SDK_LIBRARY_FILE}
			INTERFACE_INCLUDE_DIRECTORIES "${FBX_SDK_INCLUDE_DIR}"
		)
	endif()

	if(WIN32)
		install(FILES ${FBX_SDK_DLL} DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR})
	endif()
endif()

include(FindPackageHandleStandardArgs)

FIND_PACKAGE_HANDLE_STANDARD_ARGS(FBXSDK
                                  REQUIRED_VARS FBX_SDK_LIBRARY FBX_SDK_INCLUDE_DIR
                                  VERSION_VAR FBX_SDK_VERSION_STRING)
