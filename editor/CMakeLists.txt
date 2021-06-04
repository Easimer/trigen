set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

find_package(Qt5 COMPONENTS Widgets Svg REQUIRED)
find_package(FBXSDK REQUIRED)

set(SRC
    stdafx.h
    entry.cpp

    session.cpp
    session.h
    wnd_main.cpp
    wnd_main.h
    vm_main.cpp
    vm_main.h
    world.cpp
    world.h
    transform_component.h
    collider_component.h
    plant_component.h
    mesh_render_component.h
    wizard_sb_simulation.cpp
    wizard_sb_simulation.h
    wizard_sb_simulation_pages.h
    wizard_collider.cpp
    wizard_collider.h
    wizard_collider_pages.h
    collimp.h
    collimp_obj.cpp
    mesh_collider.h
    entity_list.cpp
    entity_list.h
    world_qt.cpp
    world_qt.h
    dlg_meshgen.cpp
    dlg_meshgen.h
    vm_meshgen.cpp
    vm_meshgen.h
    trigen_worker.cpp
    trigen_worker.h

    tiny_obj_loader.cpp

    wnd_main.ui
    wizard_sb_plant_config.ui
    wizard_sb_compute_config.ui
    wizard_collider_browse.ui
    dlg_meshgen.ui

    resources.qrc
)

add_executable(editor ${SRC})
target_link_libraries(editor
    PRIVATE
    Qt5::Widgets Qt5::Svg
    FbxSdk::FbxSdk 
    RCmd::Base
    arcball_camera renderer_qt
    mesh_export trigen uv_inspect
    stb
)
target_precompile_headers(editor PRIVATE "stdafx.h")
tri_builddir(editor)

add_custom_command(
    TARGET editor POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:Qt5::Core>
        $<TARGET_FILE:Qt5::Gui>
        $<TARGET_FILE:Qt5::Widgets>
        $<TARGET_FILE:Qt5::Svg>
        ${CMAKE_BINARY_DIR}
)

install(TARGETS editor)
