set(SRC
    src/impl.c
    include/stb_image.h
    include/stb_image_write.h
)

add_library(stb STATIC ${SRC})
target_include_directories(stb PUBLIC include/)
set_target_properties(stb PROPERTIES FOLDER "Libs")