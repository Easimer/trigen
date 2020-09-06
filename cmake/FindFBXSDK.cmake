# TODO: dynamic linking fbxsdk is not supported

if(NOT FBX_SDK_DIR)
	set(FBX_SDK_DIR "" CACHE PATH "FBX SDK directory (the one that contains 'include', 'lib' and 'samples'")
else()
	if(NOT FBX_SDK_BUILD_TYPE)
		# Default to using the release library, even in debug mode, but let users
		# override FBX_SDK_BUILD_TYPE if they want to
		set(FBX_SDK_BUILD_TYPE "release" CACHE STRING "FBX SDK build type")
	endif()

	if(WIN32)
		set(FBX_SDK_PLAT "vs2017")
		if(CMAKE_SIZEOF_VOID_P EQUAL 8)
			set(FBX_SDK_ARCH "x64")
		elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
			set(FBX_SDK_ARCH "x86")
		endif()

		set(FBX_SDK_LIB_ROOT "${FBX_SDK_DIR}/lib/${FBX_SDK_PLAT}/${FBX_SDK_ARCH}/${FBX_SDK_BUILD_TYPE}/")
		get_filename_component(FBX_SDK_VERSION_STRING ${FBX_SDK_DIR} NAME)

		if(MSVC_RUNTIME_LIBRARY EQUAL "MultiThreaded")
			set(FBX_SDK_LIBRARY_SUFFIX "-mt")
		elseif(MSVC_RUNTIME_LIBRARY EQUAL "MultiThreadedDLL")
			set(FBX_SDK_LIBRARY_SUFFIX "-md")
		elseif(MSVC_RUNTIME_LIBRARY EQUAL "MultiThreadedDebug")
			set(FBX_SDK_LIBRARY_SUFFIX "-mt")
		elseif(MSVC_RUNTIME_LIBRARY EQUAL "MultiThreadedDebugDLL")
			set(FBX_SDK_LIBRARY_SUFFIX "-md")
		endif()

		set(FBX_SDK_LIBRARY "${FBX_SDK_LIB_ROOT}/libfbxsdk${FBX_SDK_LIBRARY_SUFFIX}.lib")
		set(FBX_SDK_INCLUDE_DIR "${FBX_SDK_DIR}/include/")
		message("FBX SDK version: '${FBX_SDK_VERSION_STRING}' lib: ${FBX_SDK_LIBRARY} include: ${FBX_SDK_INCLUDE_DIR}")
	endif()
endif()

include(FindPackageHandleStandardArgs)

FIND_PACKAGE_HANDLE_STANDARD_ARGS(FBXSDK
                                  REQUIRED_VARS FBX_SDK_LIBRARY FBX_SDK_INCLUDE_DIR
                                  VERSION_VAR FBX_SDK_VERSION_STRING)