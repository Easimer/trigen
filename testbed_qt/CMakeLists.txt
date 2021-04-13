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

    softbody_renderer.cpp
    softbody_renderer.h

    colliders.cpp
    colliders.h

    model_export.h
    model_export_fbx.cpp

    debug_visualizer.cpp
    debug_visualizer.h

    vm_main.cpp
    vm_main.h

    resources.qrc
)

add_executable(testbed_qt ${SRC})
target_link_libraries(testbed_qt Qt5::Widgets softbody softbody_ext renderer NodeEditor::nodes marching_cubes FbxSdk::FbxSdk objscan psp arcball_camera renderer_qt RCmd::Base RCmd::Softbody)

install(TARGETS testbed_qt)

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

# https://stackoverflow.com/a/41199492
if(Qt5_FOUND AND WIN32 AND TARGET Qt5::qmake AND NOT TARGET Qt5::windeployqt)
    get_target_property(_qt5_qmake_location Qt5::qmake IMPORTED_LOCATION)

    execute_process(
        COMMAND "${_qt5_qmake_location}" -query QT_INSTALL_PREFIX
        RESULT_VARIABLE return_code
        OUTPUT_VARIABLE qt5_install_prefix
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    set(imported_location "${qt5_install_prefix}/bin/windeployqt.exe")

    if(EXISTS ${imported_location})
        add_executable(Qt5::windeployqt IMPORTED)

        set_target_properties(Qt5::windeployqt PROPERTIES
            IMPORTED_LOCATION ${imported_location}
        )
    endif()
endif()

if(TARGET Qt5::windeployqt)
    # execute windeployqt in a tmp directory after build
    add_custom_command(TARGET testbed_qt 
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E remove_directory "${CMAKE_CURRENT_BINARY_DIR}/windeployqt"
        COMMAND set PATH=%PATH%$<SEMICOLON>${qt5_install_prefix}/bin
        COMMAND Qt5::windeployqt --dir "${CMAKE_CURRENT_BINARY_DIR}/windeployqt" "$<TARGET_FILE_DIR:testbed_qt>/$<TARGET_FILE_NAME:testbed_qt>"
    )

    # copy deployment directory during installation
    install(
        DIRECTORY
        "${CMAKE_CURRENT_BINARY_DIR}/windeployqt/"
        DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}
    )
endif()

install(TARGETS nodes DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR})
