set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

find_package(Qt5 COMPONENTS Widgets REQUIRED)

set(SRC
    src/glviewport.cpp
    src/imgui_impl_qt.cpp
    include/imgui_impl_qt.h
    include/glviewport.h
)

add_library(renderer_qt STATIC ${SRC})
target_link_libraries(renderer_qt PRIVATE Qt5::Widgets renderer arcball_camera)
target_include_directories(renderer_qt PUBLIC include/)
#target_precompile_headers(renderer_qt PRIVATE "src/stdafx.h")
set_target_properties(renderer_qt PROPERTIES FOLDER "Libs")