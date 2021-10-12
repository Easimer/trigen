// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: 
//

#include <algorithm>

#include "gl_texture_manager.h"

namespace topo {
bool
GL_Texture_Manager::CreateTexture(
    Texture_ID *outHandle,
    unsigned width,
    unsigned height,
    Texture_Format format,
    void const *image) {
    // Ptr to image data may be NULL only if the image is empty.
    if (image == nullptr && (width != 0 || height != 0)) {
        return false;
    }

    if (outHandle == nullptr) {
        return false;
    }

    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    // Set up wrapping behavior
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // Set up filtering
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // Set up mipmap generation
    glTexParameteri(
        GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    switch (format) {
    case Texture_Format::RGB888:
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB,
            GL_UNSIGNED_BYTE, image);
        break;
    case Texture_Format::SRGB888:
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_SRGB, width, height, 0, GL_RGB,
            GL_UNSIGNED_BYTE, image);
        break;
    case Texture_Format::RGBA8888:
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_SRGB8_ALPHA8, width, height, 0, GL_RGBA,
            GL_UNSIGNED_BYTE, image);
        break;
    }

    glGenerateMipmap(GL_TEXTURE_2D);

    printf("Created texture %d\n", texture);
    _textures.push_front(texture);
    *outHandle = &_textures.front();

    return true;
}

void
GL_Texture_Manager::DestroyTexture(Texture_ID id) {
    if (id == nullptr) {
        return;
    }

    std::remove_if(_textures.begin(), _textures.end(), [&](GLuint const &t) {
        return &t == id;
    });
}
GLuint
GL_Texture_Manager::GetHandle(Texture_ID id) {
    return *(GLuint *)id;
}
}
