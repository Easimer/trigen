# === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
#
# Purpose: CMake module for the Tracy C++ frame profiler
#

if(NOT TRACY_SRC_DIR)
    set(TRACY_SRC_DIR "${CMAKE_SOURCE_DIR}/tracy" CACHE PATH "Path to the Tracy repository")
    message(WARNING "Using built-in version of Tracy at ${TRACY_SRC_DIR}")
endif()

add_library(tracy_client STATIC ${TRACY_SRC_DIR}/TracyClient.cpp)
target_include_directories(tracy_client PUBLIC ${TRACY_SRC_DIR})
target_compile_definitions(tracy_client PRIVATE TRACY_ENABLE=1)
set_target_properties(tracy_client PROPERTIES FOLDER "Libs")

set(TRACY_LIBRARIES tracy_client)
set(TRACY_INCLUDE_DIR ${TRACY_SRC_DIR})

add_library(Tracy::Client ALIAS tracy_client)

include(FindPackageHandleStandardArgs)

FIND_PACKAGE_HANDLE_STANDARD_ARGS(Tracy REQUIRED_VARS TRACY_LIBRARIES)
