set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

find_package(Qt5 COMPONENTS Widgets REQUIRED)
find_package(Filament REQUIRED)

set(SRC
    stdafx.h
    entry.cpp

    session.cpp
    session.h
    renderer.cpp
    renderer.h
    wnd_main.cpp
    wnd_main.h
    filament_viewport.cpp
    filament_viewport.h
)

set(FILAMENT_LIBS
    Filament::filament
    Filament::backend
    Filament::bluegl
    Filament::bluevk
    Filament::filabridge
    Filament::filaflat
    Filament::utils
    Filament::geometry
    Filament::smol-v
    Filament::vkshaders
    Filament::ibl
)

filament_matc(${CMAKE_CURRENT_SOURCE_DIR}/materials/bakedColor.mat bakedColor.matc)
embed_file(bakedColor.matc.cpp bakedColor.matc)

add_executable(editor ${SRC} bakedColor.matc.cpp)
target_link_libraries(editor PRIVATE softbody Qt5::Widgets arcball_camera ${FILAMENT_LIBS})
target_precompile_headers(editor PRIVATE "stdafx.h")
tri_builddir(editor)

