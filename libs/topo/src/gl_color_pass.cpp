// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//


#include "gl_color_pass.h"

#include "glres.h"
#include "shader_program_builder.h"

extern "C" {
extern char const *solid_color_vsh_glsl;
extern char const *solid_color_fsh_glsl;
}

namespace topo {

GL_Color_Pass::GL_Color_Pass(
    GL_Model_Manager *modelManager,
    Renderable_Manager *renderableManager,
    Material_Manager *materialManager,
    GL_Texture_Manager *textureManager,
    Shader_Generic_Textured_Unlit *shaderTexturedUnlit,
    Shader_Solid_Color *shaderSolidColor)
    : _modelManager(modelManager)
    , _renderableManager(renderableManager)
    , _shaderTexturedUnlit(shaderTexturedUnlit)
    , _shaderSolidColor(shaderSolidColor)
    , _materialManager(materialManager)
    , _textureManager(textureManager) { }

void
GL_Color_Pass::Execute(Render_Queue *renderQueue, glm::mat4 const &matVP) {
    for (auto &cmd : renderQueue->GetCommands()) {
        Model_ID model;
        Material_ID material;
        _renderableManager->GetModelAndMaterial(
            cmd.renderable, &model, &material);

        switch (_materialManager->GetType(material)) {
        case topo::MAT_UNLIT:
            RenderUnlit(model, material, cmd.transform, matVP);
            break;
        case topo::MAT_SOLID_COLOR:
            RenderSolidColor(model, material, cmd.transform, matVP);
            break;
        }
    }
}

void
GL_Color_Pass::RenderUnlit(
    Model_ID model,
    Material_ID material,
    Transform const &transform, glm::mat4 const &matVP) {
    glUseProgram(_shaderTexturedUnlit->program());

    auto matTransform = glm::translate(transform.position)
        * glm::mat4_cast(transform.rotation) * glm::scale(transform.scale);
    gl::SetUniformLocation(_shaderTexturedUnlit->locMVP(), matVP * matTransform);

    auto *materialData = (topo::Material_Unlit *)_materialManager->GetMaterialData(material);

    auto texDiffuse = _textureManager->GetHandle(materialData->diffuse);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texDiffuse);
    gl::SetUniformLocation(_shaderTexturedUnlit->locTexDiffuse(), 0);

    gl::SetUniformLocation(
        _shaderTexturedUnlit->locTintColor(), { 1, 1, 1, 1 });

    void *indexOffset;
    GLint baseVertex;
    GLenum elementType;
    size_t numElements;

    _modelManager->GetDrawParameters(
        model, &indexOffset, &baseVertex, &elementType, &numElements);

    glDrawElementsBaseVertex(
        GL_TRIANGLES, numElements, elementType, indexOffset, baseVertex);
}

void
GL_Color_Pass::RenderSolidColor(
    Model_ID model,
    Material_ID material,
    Transform const &transform, glm::mat4 const &matVP) {
    glUseProgram(_shaderSolidColor->Program());

    auto matTransform = glm::translate(transform.position)
        * glm::mat4_cast(transform.rotation) * glm::scale(transform.scale);

    auto *materialData = (topo::Material_Solid_Color *)_materialManager->GetMaterialData(material);

    gl::SetUniformLocation(
        _shaderSolidColor->locMatMVP(), matVP * matTransform);
    gl::SetUniformLocation(_shaderSolidColor->locMatModel(), matTransform);
    gl::SetUniformLocation(
        _shaderSolidColor->locColor(),
        { materialData->color[0], materialData->color[1],
          materialData->color[2] });

    void *indexOffset;
    GLint baseVertex;
    GLenum elementType;
    size_t numElements;

    _modelManager->GetDrawParameters(
        model, &indexOffset, &baseVertex, &elementType, &numElements);

    glDrawElementsBaseVertex(
        GL_TRIANGLES, numElements, elementType, indexOffset, baseVertex);
}

void
Shader_Solid_Color::Build() {
    printf("[ topo ] building shader 'solid color'\n");
    auto vsh
        = FromStringLoadShader<GL_VERTEX_SHADER>(solid_color_vsh_glsl);
    auto fsh
        = FromStringLoadShader<GL_FRAGMENT_SHADER>(solid_color_fsh_glsl);

    auto builder = gl::Shader_Program_Builder();
    auto program = builder.Attach(vsh).Attach(fsh).Link();
    if (program) {
        _program = std::move(program.value());

        _locMatMVP = gl::Uniform_Location<glm::mat4>(_program, "matMVP");
        _locMatModel = gl::Uniform_Location<glm::mat4>(_program, "matModel");
        _locColor = gl::Uniform_Location<glm::vec3>(_program, "solidColor");
    } else {
        throw Shader_Linker_Exception("solid color", builder.Error());
    }
}

}
