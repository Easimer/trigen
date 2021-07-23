// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#pragma once

#include <optional>

#include "glres.h"
#include "render_queue.h"

namespace topo {
struct G_Buffer_Shader_Program {
    gl::Shader_Program program;

    gl::Uniform_Location<GLint> texBaseColor;
    gl::Uniform_Location<GLint> texNormal;
    gl::Uniform_Location<GLint> texPosition;
    gl::Uniform_Location<GLint> texShadowMap;
    gl::Uniform_Location<glm::mat4> shadowCasterViewProj;
};

struct G_Buffer_Light {
    glm::vec3 position;
    glm::vec3 color;
};

struct G_Buffer_Draw_Params {
    glm::vec3 viewPosition;
    std::vector<G_Buffer_Light> lights;
};

class G_Buffer {
public:
    G_Buffer(char const *name, unsigned width, unsigned height);

    void
    activate();
    void
    draw(
        Render_Queue *rq,
        glm::vec3 const &viewPosition,
        GLint readFramebuffer,
        GLint drawFramebuffer,
        unsigned screenWidth,
        unsigned screenHeight,
        gl::Texture &shadowMap,
        glm::mat4 const &shadowCasterViewProj);

    gl::Framebuffer &
    GetFramebuffer() {
        return _fb;
    }

private:
    unsigned _width;
    unsigned _height;
    gl::Framebuffer _fb;
    gl::Texture _bufBaseColor;
    gl::Texture _bufNormal;
    gl::Texture _bufPosition;
    gl::Renderbuffer _bufRender;

    gl::VAO _quadVao;
    gl::VBO _quadPosVbo;
    gl::VBO _quadUvVbo;

    std::optional<G_Buffer_Shader_Program> _program;
};
}
