# === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
#
# Purpose: CMake module for the Filament renderer
#

# Parameters:
# - FILAMENT_DIR:PATH = path to the pre-built Filament binaries (the
#   directory that contains bin, docs, include and lib)

macro(filament_add_library TARGET LINKDIR INCLUDEDIR)
    add_library("Filament::${TARGET}" INTERFACE IMPORTED)
    set_target_properties("Filament::${TARGET}" PROPERTIES
        IMPORTED_LIBNAME "${TARGET}"
        INTERFACE_LINK_DIRECTORIES "${LINKDIR}"
        INTERFACE_INCLUDE_DIRECTORIES "${INCLUDEDIR}"
    )
endmacro()

if(NOT FILAMENT_DIR)
    set(FILAMENT_DIR "" CACHE PATH "Path to the pre-built Filament binaries (the directory that contains bin, docs, include and lib)")
    message(FATAL_ERROR "Please set FILAMENT_DIR")
else()
    set(LINKDIR "${FILAMENT_DIR}/lib/")
    set(INCLUDEDIR "${FILAMENT_DIR}/include/")

    filament_add_library(filament ${LINKDIR} ${INCLUDEDIR})
    filament_add_library(backend ${LINKDIR} ${INCLUDEDIR})
    filament_add_library(bluegl ${LINKDIR} ${INCLUDEDIR})
    filament_add_library(bluevk ${LINKDIR} ${INCLUDEDIR})
    filament_add_library(filabridge ${LINKDIR} ${INCLUDEDIR})
    filament_add_library(filaflat ${LINKDIR} ${INCLUDEDIR})
    filament_add_library(utils ${LINKDIR} ${INCLUDEDIR})
    filament_add_library(geometry ${LINKDIR} ${INCLUDEDIR})
    filament_add_library(smol-v ${LINKDIR} ${INCLUDEDIR})
    filament_add_library(vkshaders ${LINKDIR} ${INCLUDEDIR})
    filament_add_library(ibl ${LINKDIR} ${INCLUDEDIR})

    if(WIN32)
        set_target_properties(Filament::bluegl PROPERTIES INTERFACE_LINK_LIBRARIES Opengl32)
    endif()

    set(FILAMENT_FOUND "yes")
endif()

include(FindPackageHandleStandardArgs)

FIND_PACKAGE_HANDLE_STANDARD_ARGS(
    Filament
    REQUIRED_VARS FILAMENT_FOUND 
)
