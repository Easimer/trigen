set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

find_package(Qt5 COMPONENTS Widgets REQUIRED)
find_package(FBXSDK REQUIRED)

set(SRC
    common.h
    main.cpp

    sim_control.ui
    sim_config.ui

    helper_ui_widget.h

    wnd_main.cpp
    wnd_main.h

    wnd_meshgen.cpp
    wnd_meshgen.h

    glviewport.cpp
    glviewport.h

    arcball_camera.cpp
    arcball_camera.h

    softbody_renderer.cpp
    softbody_renderer.h

    colliders.cpp
    colliders.h

    file_serializer.cpp

    model_export.h
    model_export_fbx.cpp

    resources.qrc
)

add_executable(testbed_qt ${SRC})
target_link_libraries(testbed_qt Qt5::Widgets softbody renderer NodeEditor::nodes tree_meshifier ${FBX_SDK_LIBRARY})
target_include_directories(testbed_qt PRIVATE ${FBX_SDK_INCLUDE_DIR})

add_custom_command(
    TARGET testbed_qt POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:Qt5::Core>
        $<TARGET_FILE:Qt5::Gui>
        $<TARGET_FILE:Qt5::Widgets>
        $<TARGET_FILE:NodeEditor::nodes>
        ${CMAKE_BINARY_DIR}
)

target_precompile_headers(testbed_qt PRIVATE "common.h")
