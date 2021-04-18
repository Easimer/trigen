set(SRC_IMGUI
	imgui.cpp
	imgui_draw.cpp
	imgui_impl_opengl3.cpp
	imgui_impl_sdl.cpp
	imgui_widgets.cpp
	ImGuizmo.cpp

	imgui_internal.h
	imstb_rectpack.h
	imstb_textedit.h
	imstb_truetype.h

	../public/imgui.h
	../public/imgui_impl_opengl3.h
	../public/imgui_impl_sdl.h
	../public/imconfig.h
	../public/ImGuizmo.h
)

add_library(imgui STATIC ${SRC_IMGUI})
target_include_directories(imgui PRIVATE ${SDL2_INCLUDE_DIRS})
target_link_libraries(imgui PRIVATE glad ${SDL2_LIBRARIES})
target_compile_definitions(imgui PRIVATE IMGUI_IMPL_OPENGL_LOADER_GLAD)
set_target_properties(imgui PROPERTIES FOLDER "Libs")

target_disable_clang_tidy(imgui)