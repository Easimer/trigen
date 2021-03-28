find_package(Filament REQUIRED)

set(SRC
	stdafx.h
	entry.cpp
)

add_executable(editor ${SRC})
target_link_libraries(editor PRIVATE softbody Filament::filament)
target_precompile_headers(editor PRIVATE "stdafx.h")
tri_builddir(editor)
