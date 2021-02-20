set(SRC
    src/file_serializer.cpp
    include/softbody/file_serializer.h
)

add_library(softbody_ext STATIC ${SRC})
target_include_directories(softbody_ext PUBLIC include/)
set_target_properties(softbody_ext PROPERTIES FOLDER "Libs")
