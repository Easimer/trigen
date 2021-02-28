set(SRC
    stdafx.h
    main.cpp
)

add_executable(embed ${SRC})
target_precompile_headers(embed PRIVATE "stdafx.h")

macro(embed_file storage_file)
    set(source_files ${ARGN})

    list(LENGTH source_files source_file_count)
    if(${source_file_count} GREATER 0)
        add_custom_command(
            OUTPUT ${storage_file}
            COMMAND embed ${storage_file} ${source_files}
            DEPENDS ${source_files})
		if(MSVC)
			set_source_files_properties(${storage_file}
				PROPERTIES COMPILE_FLAGS "/bigobj"
			)
        endif()
    else()
        warning("embed_file: no input files")
    endif()
endmacro()
