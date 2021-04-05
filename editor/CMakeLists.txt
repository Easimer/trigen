set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

find_package(Qt5 COMPONENTS Widgets REQUIRED)
find_package(Filament REQUIRED)
find_package(FBXSDK REQUIRED)

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
    world.h
    world_object.h
    world_renderable.h
    world_collider.h
    world_plant.h
    filament_wrapper.h
    wizard_sb_simulation.cpp
    wizard_sb_simulation.h
    wizard_sb_simulation_pages.h

    wnd_main.ui
    wizard_sb_plant_config.ui
    wizard_sb_compute_config.ui
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
target_link_libraries(editor PRIVATE softbody Qt5::Widgets arcball_camera FbxSdk::FbxSdk ${FILAMENT_LIBS})
target_precompile_headers(editor PRIVATE "stdafx.h")
tri_builddir(editor)

