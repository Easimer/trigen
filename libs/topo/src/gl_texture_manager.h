// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: 
//

#pragma once

#include <list>

#include <topo.h>

#include <glad/glad.h>

namespace topo {

class GL_Texture_Manager {
public:
    bool
    CreateTexture(
        Texture_ID *outHandle,
        unsigned width,
        unsigned height,
        Texture_Format format,
        void const *image);

    void
    DestroyTexture(Texture_ID texture);

    GLuint GetHandle(Texture_ID texture);

private:
    std::list<GLuint> _textures;
};

}
