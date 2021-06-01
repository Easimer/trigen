// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#pragma once

#include <optional>

#include "glres.h"

struct G_Buffer_Shader_Program {
    gl::Shader_Program program;
    gl::Uniform_Location<GLint> texBaseColor;
    gl::Uniform_Location<GLint> texNormal;
    gl::Uniform_Location<GLint> texPosition;
};

class G_Buffer {
public:
    G_Buffer();
    G_Buffer(unsigned width, unsigned height);

    void activate();
    void draw();
private:
    gl::Framebuffer _fb;
    gl::Texture _bufBaseColor;
    gl::Texture _bufNormal;
    gl::Texture _bufPosition;
    gl::Renderbuffer _bufRender;

    gl::VAO _quadVao;
    gl::VBO _quadPosVbo;
    gl::VBO _quadUvVbo;

    std::optional<G_Buffer_Shader_Program> _program;

    GLint _prevFBRead;
    GLint _prevFBDraw;
};
