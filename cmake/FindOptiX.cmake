# === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
#
# Purpose: CMake module for the NVIDIA OptiX 7 SDK
#


if(NOT OPTIX_DIR)
	set(OPTIX_DIR "" CACHE PATH "Path to OptiX installation directory")
else()
	find_path(OPTIX_INCLUDE_DIR NAMES optix.h PATHS "${OPTIX_DIR}/include/" NO_DEFAULT_PATH)
	find_path(OPTIX_INCLUDE_DIR NAMES optix.h)
endif()
