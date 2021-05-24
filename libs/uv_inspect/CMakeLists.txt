add_library(uv_inspect SHARED
    src/uv_inspect.cpp
    include/uv_inspect.hpp
)
target_include_directories(uv_inspect
    PUBLIC
        include/
    PRIVATE
		${SDL2_INCLUDE_DIRS}
)
target_link_libraries(uv_inspect
	PUBLIC
        psp
    PRIVATE
        ${SDL2_LIBRARIES}
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