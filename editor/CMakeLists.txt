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
    renderer.cpp
    renderer.h
    wnd_main.cpp
    wnd_main.h
    vm_main.cpp
    vm_main.h
    world.cpp
    world.h
    collider_component.h
    plant_component.h
    wizard_sb_simulation.cpp
    wizard_sb_simulation.h
    wizard_sb_simulation_pages.h

    wnd_main.ui
    wizard_sb_plant_config.ui
    wizard_sb_compute_config.ui

    resources.qrc
)

add_executable(editor ${SRC})
target_link_libraries(editor PRIVATE softbody Qt5::Widgets Qt5::Svg arcball_camera FbxSdk::FbxSdk renderer_qt)
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
