add_library(uv_inspect SHARED
    src/uv_inspect.cpp
    include/uv_inspect.hpp
)
target_link_directories(uv_inspect
	PRIVATE
		${SDL2TTF_LIBRARY}
)
target_include_directories(uv_inspect
    PUBLIC
        include/
    PRIVATE
		${SDL2_INCLUDE_DIRS}
        ${SDL2TTF_INCLUDE_DIR}
)
target_link_libraries(uv_inspect
	PUBLIC
        psp
    PRIVATE
        ${SDL2_LIBRARIES}
        SDL2_ttf
)
set_target_properties(uv_inspect
    PROPERTIES
        FOLDER "Libs"
)
target_compile_definitions(uv_inspect
    PRIVATE
        UV_INSPECTOR_BUILDING=1
)


tri_dllimportexport(uv_inspect)
tri_builddir(uv_inspect)

install(TARGETS uv_inspect)