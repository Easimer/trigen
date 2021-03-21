find_package(FBXSDK REQUIRED)

set(SRC
    src/stdafx.h
    src/fbx_export.cpp
    include/mesh_export.h
    src/fbx_standard_surface.cpp
    src/fbx_standard_surface.h
)

add_library(mesh_export STATIC ${SRC})
target_link_libraries(mesh_export PRIVATE psp stb FbxSdk::FbxSdk)
target_include_directories(mesh_export PUBLIC include/)
target_precompile_headers(mesh_export PRIVATE "src/stdafx.h")