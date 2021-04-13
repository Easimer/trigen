set(SRC_GENERAL
    src/r_general.cpp
    include/r_cmd/general.h
)

set(SRC_SOFTBODY
    src/r_softbody.cpp
    include/r_cmd/softbody.h
)

macro(add_render_command_library NAME SOURCES DEPS)
    add_library(r_cmd_${NAME} STATIC ${SOURCES})
    target_link_libraries(r_cmd_${NAME} PRIVATE renderer ${DEPS})
    target_include_directories(r_cmd_${NAME} PUBLIC include/)
    target_precompile_headers(r_cmd_${NAME} PRIVATE "src/stdafx.h")
    set_target_properties(r_cmd_${NAME} PROPERTIES FOLDER "Libs/Render Commands")
    add_library(RCmd::${NAME} ALIAS r_cmd_${NAME})
endmacro()

add_render_command_library(Base "${SRC_GENERAL}" "stb")
add_render_command_library(Softbody "${SRC_SOFTBODY}" "softbody")
