// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#pragma once

#include <topo.h>

#include <unordered_map>

#include "gl_model_manager.h"
#include "gl_multidraw.h"
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
    locMatVP() {
        return _locMatVP;
    }

    gl::Uniform_Location<glm::vec3> &
    locColor() {
        return _locColor;
    }

private:
    gl::Shader_Program _program;

    gl::Uniform_Location<glm::mat4> _locMatVP;
    gl::Uniform_Location<glm::vec3> _locColor;
};

class Shader_Lines {
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

    gl::Uniform_Location<glm::vec3> &
    locColor() {
        return _locColor;
    }

private:
    gl::Shader_Program _program;
    gl::Uniform_Location<glm::mat4> _locMatMVP;
    gl::Uniform_Location<glm::vec3> _locColor;
};

class Shader_Textured_Unlit {
public:
    void
    Build();

    gl::Shader_Program &
    Program() {
        return _program;
    }

    gl::Uniform_Location<glm::mat4> &
    locMatVP() {
        return _locMatVP;
    }

    gl::Uniform_Location<GLint>
    locTexDiffuse() {
        return _locTexDiffuse;
    }

private:
    gl::Shader_Program _program;

    gl::Uniform_Location<glm::mat4> _locMatVP;
    gl::Uniform_Location<GLint> _locTexDiffuse;
};

class Shader_Textured_Transparent {
public:
    void
    Build();

    gl::Shader_Program &
    Program() {
        return _program;
    }

    gl::Uniform_Location<glm::mat4> &
    locMatVP() {
        return _locMatVP;
    }

    gl::Uniform_Location<GLint>
    locTexDiffuse() {
        return _locTexDiffuse;
    }

private:
    gl::Shader_Program _program;

    gl::Uniform_Location<glm::mat4> _locMatVP;
    gl::Uniform_Location<GLint> _locTexDiffuse;
};

class Shader_Textured_Lit {
public:
    void
    Build();

    gl::Shader_Program &
    Program() {
        return _program;
    }

    gl::Uniform_Location<glm::mat4> &
    locMatVP() {
        return _locMatVP;
    }

    gl::Uniform_Location<GLint>
    locTexDiffuse() {
        return _locTexDiffuse;
    }

    gl::Uniform_Location<GLint>
    locTexNormal() {
        return _locTexNormal;
    }

private:
    gl::Shader_Program _program;

    gl::Uniform_Location<glm::mat4> _locMatVP;
    gl::Uniform_Location<GLint> _locTexDiffuse;
    gl::Uniform_Location<GLint> _locTexNormal;
};

class GL_Color_Pass {
public:
    GL_Color_Pass(
        GL_Model_Manager *modelManager,
        Renderable_Manager *renderableManager,
        Material_Manager *materialManager,
        GL_Texture_Manager *textureManager,
        Shader_Textured_Unlit *shaderTexturedUnlit,
        Shader_Textured_Transparent *shaderTexturedTransparent,
        Shader_Textured_Lit *shaderTexturedLit,
        Shader_Solid_Color *shaderSolidColor,
        Shader_Lines *shaderLines);

    void
    Execute(Render_Queue *renderQueue, GL_Multidraw &multiDraw, glm::mat4 const &matVP);

protected:
    struct Instance {
        Model_ID model;
        Transform transform;
    };

    using Material_Instances = std::unordered_map<Material_ID, std::vector<Instance>>;

    void
    RenderModels(
        GL_Multidraw &mutliDraw,
        glm::mat4 const &matVP);

    void
    RenderLines(
        std::vector<Render_Queue::Command> const &commands,
        glm::mat4 const &matVP);

private:
    GL_Model_Manager *_modelManager;
    Renderable_Manager *_renderableManager;
    Material_Manager *_materialManager;
    GL_Texture_Manager *_textureManager;

    Shader_Textured_Unlit *_shaderTexturedUnlit;
    Shader_Textured_Transparent *_shaderTexturedTransparent;
    Shader_Textured_Lit *_shaderTexturedLit;
    Shader_Solid_Color *_shaderSolidColor;
    Shader_Lines *_shaderLines;
};
}
