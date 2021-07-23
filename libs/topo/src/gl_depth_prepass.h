// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#pragma once

#include <topo.h>

#include "gl_model_manager.h"
#include "gl_multidraw.h"
#include "glres.h"
#include "render_queue.h"
#include "renderable_manager.h"

namespace topo {

class GL_Depth_Pass_Shader {
public:
    void
    build();

    gl::Shader_Program &
    Program() {
        return _program;
    }

    gl::Uniform_Location<glm::mat4> &
    locVP() {
        return _locVP;
    }

private:
    gl::Shader_Program _program;
    gl::Uniform_Location<glm::mat4> _locVP;
};

class GL_Depth_Pass {
public:
    GL_Depth_Pass(std::string const &name, GL_Model_Manager *modelManager, Renderable_Manager *renderableManager, unsigned width, unsigned height, GL_Depth_Pass_Shader *shader);

    void
    Clear();

    void
    Execute(Render_Queue *rq, GL_Multidraw &multiDraw, glm::mat4 matVP);

    void
    BlitDepth(GLuint destFramebuffer);

    gl::Texture &
    DepthMap() {
        return _depthMap;
    }

private:
    GL_Model_Manager *_modelManager;
    GL_Depth_Pass_Shader *_shader;
    Renderable_Manager *_renderableManager;

    gl::Framebuffer _framebuffer;
    gl::Texture _depthMap;

    std::string _name;
    unsigned _width, _height;
};
}
