set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

find_package(Qt5 COMPONENTS Widgets REQUIRED)

set(SRC
    src/glviewport.cpp
    src/imgui_impl_qt.cpp
    include/topo/imgui_impl_qt.h
    include/topo/glviewport.h
)

add_library(topo_qt STATIC ${SRC})
target_link_libraries(topo_qt
    PUBLIC
        topo imgui
    PRIVATE
        Qt5::Widgets arcball_camera Tracy::Client
)
target_include_directories(topo_qt PUBLIC include/)
set_target_properties(topo_qt PROPERTIES FOLDER "Libs")