// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#pragma once

#include <topo.h>

#include "gl_model_manager.h"
#include "gl_texture_manager.h"
#include "glres.h"
#include "material_manager.h"
#include "render_queue.h"
#include "renderable_manager.h"
#include "shader_generic.h"

namespace topo {

class Shader_Solid_Color {
public:
    void
    Build();

    gl::Shader_Program &
    Program() {
        return _program;
    }

    gl::Uniform_Location<glm::mat4> &
    locMatMVP() {
        return _locMatMVP;
    }

    gl::Uniform_Location<glm::mat4> &
    locMatModel() {
        return _locMatModel;
    }

    gl::Uniform_Location<glm::vec3> &
    locColor() {
        return _locColor;
    }

private:
    gl::Shader_Program _program;

    gl::Uniform_Location<glm::mat4> _locMatMVP;
    gl::Uniform_Location<glm::mat4> _locMatModel;
    gl::Uniform_Location<glm::vec3> _locColor;
};

class GL_Color_Pass {
public:
    GL_Color_Pass(
        GL_Model_Manager *modelManager,
        Renderable_Manager *renderableManager,
        Material_Manager *materialManager,
        GL_Texture_Manager *textureManager,
        Shader_Generic_Textured_Unlit *shaderTexturedUnlit,
        Shader_Solid_Color *shaderSolidColor);

    void
    Execute(Render_Queue *renderQueue, glm::mat4 const &matVP);

protected:
    void
    RenderUnlit(
        Model_ID model,
        Material_ID material,
        Transform const &transform,
        glm::mat4 const &matVP);

    void
    RenderSolidColor(
        Model_ID model,
        Material_ID material,
        Transform const &transform,
        glm::mat4 const &matVP);

private:
    GL_Model_Manager *_modelManager;
    Renderable_Manager *_renderableManager;
    Material_Manager *_materialManager;
    GL_Texture_Manager *_textureManager;

    Shader_Generic_Textured_Unlit *_shaderTexturedUnlit;
    Shader_Generic_Textured_Lit *_shaderTexturedLit;
    Shader_Solid_Color *_shaderSolidColor;
};
}
