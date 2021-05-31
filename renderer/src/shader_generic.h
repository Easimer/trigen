// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: generic shader
//

#pragma once

#include "glres.h"
#include "r_gl_shadercompiler.h"

#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>

class Shader_Linker_Exception {
public:
    Shader_Linker_Exception(std::string programName, std::string errorMessage)
        : _programName(std::move(programName))
        , _errorMessage(std::move(errorMessage)) {
    }

    std::string const &programName() const { return _programName; }
    std::string const &errorMessage() const { return _errorMessage; }

private:
    std::string _programName;
    std::string _errorMessage;
};

class Shader_Generic_Base {
public:
    void build();
    virtual char const *name() const = 0;
    virtual Shader_Define_List defines() const = 0;

    gl::Shader_Program const &program() const { return _program; }
    gl::Uniform_Location<glm::mat4> locMVP() { return _locMVP; }
    gl::Uniform_Location<glm::vec4> locTintColor() { return _locTintColor; }

protected:
    virtual void linkUniforms();

private:
    gl::Shader_Program _program;
    gl::Uniform_Location<glm::mat4> _locMVP;
    gl::Uniform_Location<glm::vec4> _locTintColor;
};

class Shader_Generic : public Shader_Generic_Base {
public:
    char const *name() const override;
    Shader_Define_List defines() const override;
};

class Shader_Generic_With_Vertex_Colors : public Shader_Generic {
public:
    char const *name() const override;
    Shader_Define_List defines() const override;
};

class Shader_Generic_Textured_Unlit : public Shader_Generic {
public:
    char const *name() const override;
    Shader_Define_List defines() const override;

    gl::Uniform_Location<GLint> locTexDiffuse() { return _locTexDiffuse; }

protected:
    void linkUniforms() override;

private:
    gl::Uniform_Location<GLint> _locTexDiffuse;
};

class Shader_Generic_Textured_Lit : public Shader_Generic_Textured_Unlit {
public:
    char const *name() const override;
    Shader_Define_List defines() const override;

    gl::Uniform_Location<GLint> locTexNormal() { return _locTexNormal; }
    gl::Uniform_Location<glm::vec3> locSunPosition() { return _locSunPosition; }
    gl::Uniform_Location<glm::mat4> locModelMatrix() { return _locModelMatrix; }
    gl::Uniform_Location<glm::vec3> locViewPosition() { return _locViewPosition; }

protected:
    void linkUniforms() override;

private:
    gl::Uniform_Location<GLint> _locTexNormal;
    gl::Uniform_Location<glm::vec3> _locSunPosition;
    gl::Uniform_Location<glm::mat4> _locModelMatrix;
    gl::Uniform_Location<glm::vec3> _locViewPosition;
};
