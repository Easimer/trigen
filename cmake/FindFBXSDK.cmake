# === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
#
# Purpose: CMake module for the Autodesk FBX SDK
#

# Parameters:
# - FBX_SDK_DIR:PATH = path to the Autodesk FBX SDK installation
# - FBX_SDK_PRINT_DEBUG_MESSAGE:BOOL = makes this module dump all paths after configuration
# Usage:
# - Call find_package(FBXSDK [REQUIRED])
# - Link against target FbxSdk::FbxSdk

# TODO: dynamic linking fbxsdk is not supported

if(NOT FBX_SDK_DIR)
	set(FBX_SDK_DIR "" CACHE PATH "FBX SDK directory (the one that contains 'include', 'lib' and 'samples'")
    message(FATAL_ERROR "Please set FBX_SDK_DIR")
else()
	if(FBX_SDK_BUILD_TYPE)
        message(WARNING "Parameter FBX_SDK_BUILD_TYPE is deprecated, unsetting")
		unset(FBX_SDK_BUILD_TYPE)
	endif()

    # Create imported target
	if(NOT TARGET FbxSdk::FbxSdk)
		add_library(FbxSdk::FbxSdk UNKNOWN IMPORTED)
	endif()

    # Here we determine:
    # - FBX_SDK_PLAT: platform (e.g. vs20xx or gcc)
    # - FBX_SDK_LIBRARY_EXT: extension of the static/link libraries (.lib or .a)
    # - FBX_SDK_DLL_EXT: extension of the dynamic libraries (.dll or .so)
    # - FBX_SDK_LIBRARY_SUFFIX: library suffix (only on Windows, -mt or -md)
    # - FBX_SDK_LIBRARY_DEPS: list of the interface dependencies of libfbxsdk
    # - FBX_SDK_INCLUDE_DIR: list of the interface dependencies' include paths

	if(WIN32)
		set(FBX_SDK_PLAT "vs2019")
		set(FBX_SDK_LIBRARY_EXT ".lib")
		set(FBX_SDK_DLL_EXT ".dll")

		# TODO(danielm): Windows platform - non-MSVC compiler support
		if(MSVC)
			if(MSVC_TOOLSET_VERSION EQUAL 140)
				set(FBX_SDK_PLAT "vs2015")
			elseif(MSVC_TOOLSET_VERSION EQUAL 141)
				set(FBX_SDK_PLAT "vs2017")
			elseif(MSVC_TOOLSET_VERSION EQUAL 142)
				set(FBX_SDK_PLAT "vs2019")
			endif()
		endif()

		if(MSVC_RUNTIME_LIBRARY EQUAL "MultiThreaded")
			set(FBX_SDK_LIBRARY_SUFFIX "-mt")
		elseif(MSVC_RUNTIME_LIBRARY EQUAL "MultiThreadedDLL")
			set(FBX_SDK_LIBRARY_SUFFIX "-md")
		elseif(MSVC_RUNTIME_LIBRARY EQUAL "MultiThreadedDebug")
			set(FBX_SDK_LIBRARY_SUFFIX "-mt")
		elseif(MSVC_RUNTIME_LIBRARY EQUAL "MultiThreadedDebugDLL")
			set(FBX_SDK_LIBRARY_SUFFIX "-md")
		else()
			# MSVC_RUNTIME_LIBRARY can be an empty string, in which case the
			# generator chooses a default library
			set(FBX_SDK_LIBRARY_SUFFIX "-md")
		endif()

        # This would be the paths to zlib-xx.lib and libxml2-xx.lib but we can't 
        # build those paths until later on
        # See :InterfaceDepPathsWin32
		set(FBX_SDK_LIBRARY_DEPS "")
	endif(WIN32)

	if(UNIX AND NOT APPLE)
		set(FBX_SDK_PLAT "gcc")
		set(FBX_SDK_LIBRARY_EXT ".a")
		set(FBX_SDK_DLL_EXT ".so")
		# The Linux SDK doesn't come with libxml and zlib
		find_package(ZLIB REQUIRED)
		find_package(LibXml2 REQUIRED)
		list(APPEND FBX_SDK_LIBRARY_DEPS ${ZLIB_LIBRARIES} ${LIBXML2_LIBRARIES})
	endif(UNIX AND NOT APPLE)

    # :DetermineArch
	if(CMAKE_SIZEOF_VOID_P EQUAL 8)
		set(FBX_SDK_ARCH "x64")
	elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
		set(FBX_SDK_ARCH "x86")
	endif()

    # :DetermineLibraryRootPath
    # Determine library root path based on build type
    set(FBX_SDK_LIB_ROOT ${FBX_SDK_DIR}/lib/${FBX_SDK_PLAT}/${FBX_SDK_ARCH}/debug)
    set(FBX_SDK_LIB_ROOT_RELEASE ${FBX_SDK_DIR}/lib/${FBX_SDK_PLAT}/${FBX_SDK_ARCH}/release)

    # :DetermineVersion
	get_filename_component(FBX_SDK_VERSION_STRING ${FBX_SDK_DIR} NAME)

    # :InterfaceDepPathsWin32
    # Generate the path to the included zlib and libxml2 libraries (on Windows only)
	if(WIN32)
        set(FBX_SDK_ZLIB_PATH "$<IF:$<CONFIG:Debug>,${FBX_SDK_LIB_ROOT},${FBX_SDK_LIB_ROOT_RELEASE}>/zlib${FBX_SDK_LIBRARY_SUFFIX}.lib")
        set(FBX_SDK_XML2_PATH "$<IF:$<CONFIG:Debug>,${FBX_SDK_LIB_ROOT},${FBX_SDK_LIB_ROOT_RELEASE}>/libxml2${FBX_SDK_LIBRARY_SUFFIX}.lib")
        list(APPEND FBX_SDK_LIBRARY_DEPS ${FBX_SDK_ZLIB_PATH})
        list(APPEND FBX_SDK_LIBRARY_DEPS ${FBX_SDK_XML2_PATH})
	endif()

    # :LibraryFilePaths
    # Generate the path to the static library files
    set(FBX_SDK_LIBRARY_FILE "${FBX_SDK_LIB_ROOT}/libfbxsdk${FBX_SDK_LIBRARY_SUFFIX}${FBX_SDK_LIBRARY_EXT}" CACHE FILEPATH "Path to the debug library")
    set(FBX_SDK_LIBRARY_FILE_RELEASE "${FBX_SDK_LIB_ROOT_RELEASE}/libfbxsdk${FBX_SDK_LIBRARY_SUFFIX}${FBX_SDK_LIBRARY_EXT}" CACHE FILEPATH "Path to the release library")

    # :IncludeDir
    set(FBX_SDK_INCLUDE_DIR "${FBX_SDK_DIR}/include/" CACHE PATH "FBX SDK include directory")

    # :TargetProperties
    set_target_properties(FbxSdk::FbxSdk PROPERTIES
        IMPORTED_LOCATION "${FBX_SDK_LIBRARY_FILE_RELEASE}"
        IMPORTED_LOCATION_DEBUG "${FBX_SDK_LIBRARY_FILE}"
        INTERFACE_INCLUDE_DIRECTORIES "${FBX_SDK_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES "${FBX_SDK_LIBRARY_DEPS}"
    )

    # :DLLInstall
    # Path to the DLLs so we can install them
    set(FBX_SDK_DLL "${FBX_SDK_LIB_ROOT}/libfbxsdk${FBX_SDK_LIBRARY_SUFFIX}${FBX_SDK_DLL_EXT}" CACHE FILEPATH "Path to the debug DLL")
    set(FBX_SDK_DLL_RELEASE "${FBX_SDK_LIB_ROOT_RELEASE}/libfbxsdk${FBX_SDK_LIBRARY_SUFFIX}${FBX_SDK_DLL_EXT}" CACHE FILEPATH "Path to the release DLL")

    install(FILES ${FBX_SDL_DLL} CONFIGURATIONS Debug DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR})
    install(FILES ${FBX_SDL_DLL} CONFIGURATIONS Release RelWithDebInfo MinSizeRel DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR})

    # :DebugMessage
    if(FBX_SDK_PRINT_DEBUG_MESSAGE)
        message(STATUS "=================")
        message(STATUS "FBX SDK")
        message(STATUS "Version: ${FBX_SDK_VERSION_STRING}")
        message(STATUS "Libraries: ${FBX_SDK_LIBRARY_FILE} ${FBX_SDK_LIBRARY_FILE_RELEASE}")
        message(STATUS "Include directories: ${FBX_SDK_INCLUDE_DIR}")
        message(STATUS "=================")
    endif()

    set(FBX_SDK_FOUND "yes")
endif()

include(FindPackageHandleStandardArgs)

FIND_PACKAGE_HANDLE_STANDARD_ARGS(
    FBXSDK
    REQUIRED_VARS FBX_SDK_FOUND
    VERSION_VAR FBX_SDK_VERSION_STRING
)
