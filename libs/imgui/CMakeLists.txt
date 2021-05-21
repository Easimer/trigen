set(SRC_IMGUI
	src/imgui.cpp
	src/imgui_draw.cpp
	src/imgui_impl_opengl3.cpp
	src/imgui_impl_sdl.cpp
	src/imgui_widgets.cpp
	src/ImGuizmo.cpp

	src/imgui_internal.h
	src/imstb_rectpack.h
	src/imstb_textedit.h
	src/imstb_truetype.h

	include/imgui.h
	include/imgui_impl_opengl3.h
	include/imgui_impl_sdl.h
	include/imconfig.h
	include/ImGuizmo.h
)

add_library(imgui STATIC ${SRC_IMGUI})
target_include_directories(imgui PUBLIC include/)
target_include_directories(imgui PRIVATE ${SDL2_INCLUDE_DIRS})
target_link_libraries(imgui PRIVATE glad ${SDL2_LIBRARIES})
target_compile_definitions(imgui PRIVATE IMGUI_IMPL_OPENGL_LOADER_GLAD)
set_target_properties(imgui PROPERTIES FOLDER "Libs")

target_disable_clang_tidy(imgui)