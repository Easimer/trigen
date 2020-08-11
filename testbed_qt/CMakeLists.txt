set(SOFTBODY_TESTBED_QT OFF CACHE BOOL "Build the Qt5-based testbed")

if(${SOFTBODY_TESTBED_QT})

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

find_package(Qt5 COMPONENTS Widgets REQUIRED)

set(SRC
    common.h
    main.cpp

    sim_control.ui
    sim_config.ui

    mainwindow.cpp
    mainwindow.h

    glviewport.cpp
    glviewport.h

    arcball_camera.cpp
    arcball_camera.h

    softbody_renderer.cpp
    softbody_renderer.h

    colliders.h

    resources.qrc

    ui/collider_frame.ui
    ui/collider_sphere.ui
)

add_executable(testbed_qt ${SRC})
target_link_libraries(testbed_qt Qt5::Widgets softbody renderer)

add_custom_command(
    TARGET testbed_qt POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:Qt5::Core>
        $<TARGET_FILE:Qt5::Gui>
        $<TARGET_FILE:Qt5::Widgets>
        ${CMAKE_BINARY_DIR}
)

target_precompile_headers(testbed_qt PRIVATE "common.h")

endif()
