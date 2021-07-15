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
extern char const *lines2_vsh_glsl;
extern char const *lines2_fsh_glsl;

extern char const *textured_unlit_vsh_glsl;
extern char const *textured_unlit_fsh_glsl;
}

namespace topo {

struct GL_Indirect {
    GLuint count;
    GLuint instanceCount;
    GLuint firstIndex;
    GLuint baseVertex;
    GLuint baseInstance;
};

GL_Color_Pass::GL_Color_Pass(
    GL_Model_Manager *modelManager,
    Renderable_Manager *renderableManager,
    Material_Manager *materialManager,
    GL_Texture_Manager *textureManager,
    Shader_Textured_Unlit *shaderTexturedUnlit,
    Shader_Solid_Color *shaderSolidColor,
    Shader_Lines *shaderLines)
    : _modelManager(modelManager)
    , _renderableManager(renderableManager)
    , _shaderTexturedUnlit(shaderTexturedUnlit)
    , _shaderSolidColor(shaderSolidColor)
    , _shaderLines(shaderLines)
    , _materialManager(materialManager)
    , _textureManager(textureManager) { }

void
GL_Color_Pass::Execute(Render_Queue *renderQueue, glm::mat4 const &matVP) {
    std::vector<Render_Queue::Command> modelCommands;
    std::vector<Render_Queue::Command> linesCommands;

    for (auto &cmd : renderQueue->GetCommands()) {
        switch (_renderableManager->GetRenderableKind(cmd.renderable)) {
        case Renderable_Manager::RENDERABLE_MODEL:
            modelCommands.push_back(cmd);
            break;
        case Renderable_Manager::RENDERABLE_LINES:
            linesCommands.push_back(cmd);
            break;
        }
    }

    RenderModels(modelCommands, matVP);
    RenderLines(linesCommands, matVP);
}

void
GL_Color_Pass::RenderModels(
    std::vector<Render_Queue::Command> const &commands,
    glm::mat4 const &matVP) {
    _modelManager->BindMegabuffer();

    Material_Instances cmdUnlit;
    Material_Instances cmdSolidColor;

    for (auto &cmd : commands) {
        assert(
            _renderableManager->GetRenderableKind(cmd.renderable)
            == Renderable_Manager::RENDERABLE_MODEL);

        Model_ID model;
        Material_ID material;
        _renderableManager->GetModelAndMaterial(
            cmd.renderable, &model, &material);

        switch (_materialManager->GetType(material)) {
        case topo::MAT_UNLIT:
            if (cmdUnlit.count(material) == 0) {
                cmdUnlit[material] = {};
            }

            cmdUnlit[material].push_back({ model, cmd.transform });
            break;
        case topo::MAT_SOLID_COLOR:
            if (cmdSolidColor.count(material) == 0) {
                cmdSolidColor[material] = {};
            }

            cmdSolidColor[material].push_back({ model, cmd.transform });
            break;
        }
    }

    glUseProgram(_shaderTexturedUnlit->Program());
    RenderUnlit(cmdUnlit, matVP);

    glUseProgram(_shaderSolidColor->Program());
    RenderSolidColor(cmdSolidColor, matVP);
}

void
GL_Color_Pass::RenderLines(
    std::vector<Render_Queue::Command> const &commands,
    glm::mat4 const &matVP) {
    glUseProgram(_shaderLines->Program());

    GLuint vaoLines;
    glGenVertexArrays(1, &vaoLines);
    glBindVertexArray(vaoLines);

    for (auto &cmd : commands) {
        assert(
            _renderableManager->GetRenderableKind(cmd.renderable)
            == Renderable_Manager::RENDERABLE_LINES);

        glm::vec3 const *endpoints;
        glm::vec3 color;
        size_t lineCount;
        _renderableManager->GetLines(
            cmd.renderable, &endpoints, &lineCount, &color);
        auto size = lineCount * 2 * sizeof(glm::vec3);

        GLuint bufPosition = 0;

        glGenBuffers(1, &bufPosition);
        glBindBuffer(GL_ARRAY_BUFFER, bufPosition);
        glBufferData(GL_ARRAY_BUFFER, size, endpoints, GL_STREAM_DRAW);

        auto matTransform = glm::translate(cmd.transform.position)
            * glm::mat4_cast(cmd.transform.rotation)
            * glm::scale(cmd.transform.scale);
        gl::SetUniformLocation(_shaderLines->locMatMVP(), matVP * matTransform);
        gl::SetUniformLocation(_shaderLines->locColor(), color);

        glVertexAttribPointer(
            0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);
        glEnableVertexAttribArray(0);

        glDrawArrays(GL_LINES, 0, 2 * lineCount);

        glDeleteBuffers(1, &bufPosition);
    }

    glDeleteVertexArrays(1, &vaoLines);
}

void
GL_Color_Pass::RenderUnlit(
    Material_Instances const &cmds,
    glm::mat4 const &matVP) {

    for (auto &kv : cmds) {
        auto material = kv.first;

        auto *materialData
            = (topo::Material_Unlit *)_materialManager->GetMaterialData(
                material);

        auto texDiffuse = _textureManager->GetHandle(materialData->diffuse);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texDiffuse);
        gl::SetUniformLocation(_shaderTexturedUnlit->locTexDiffuse(), 0);

        std::unordered_map<GLenum, std::vector<glm::mat4>> modelMatrices;
        std::unordered_map<GLenum, std::vector<GL_Indirect>>
            elementTypeIndirectMap;

        for (auto &cmd : kv.second) {
            auto &transform = cmd.transform;
            auto model = cmd.model;

            GLuint indexOffset;
            GLuint firstIndex;
            GLuint baseVertex;
            GLenum elementType;
            size_t numElements;

            _modelManager->GetDrawParameters(
                model, &indexOffset, &baseVertex, &elementType, &numElements,
                &firstIndex);

            if (elementTypeIndirectMap.count(elementType) == 0) {
                elementTypeIndirectMap[elementType] = {};
                modelMatrices[elementType] = {};
            }

            GL_Indirect indirect;
            indirect.baseInstance = 0;
            indirect.baseVertex = baseVertex;
            indirect.count = numElements;
            indirect.instanceCount = 1;
            indirect.firstIndex = firstIndex;

            elementTypeIndirectMap[elementType].push_back(indirect);

            auto matTransform = glm::translate(cmd.transform.position)
                * glm::mat4_cast(cmd.transform.rotation)
                * glm::scale(cmd.transform.scale);

            modelMatrices[elementType].push_back(matTransform);
        }

        gl::SetUniformLocation(
            _shaderTexturedUnlit->locMatVP(), matVP);

        std::vector<GLuint> buffers;

        for (auto &eti : elementTypeIndirectMap) {
            auto &modelMatricesVec = modelMatrices[eti.first];
            auto &indirects = eti.second;
            size_t cmdNext = 0;
            size_t cmdRemain = eti.second.size();

            const size_t BATCH_SIZE = 256;
            while (cmdRemain > 0) {
                auto currentBatchSize = (cmdRemain > BATCH_SIZE) ? BATCH_SIZE : cmdRemain;

            GLuint bufIndirect, bufUniformMatrices;

                glGenBuffers(1, &bufIndirect);
                glBindBuffer(GL_DRAW_INDIRECT_BUFFER, bufIndirect);

                glGenBuffers(1, &bufUniformMatrices);
                glBindBuffer(GL_UNIFORM_BUFFER, bufUniformMatrices);
                glBufferData(
                    GL_UNIFORM_BUFFER,
                    currentBatchSize * sizeof(modelMatricesVec[0]),
                    modelMatricesVec.data() + cmdNext, GL_STREAM_DRAW);
                glBindBufferBase(GL_UNIFORM_BUFFER, 0, bufUniformMatrices);
                glBindBuffer(GL_UNIFORM_BUFFER, 0);

                auto flags = GL_MAP_WRITE_BIT;

                auto size = currentBatchSize * sizeof(GL_Indirect);
                glBufferStorage(
                    GL_DRAW_INDIRECT_BUFFER, size,
                    nullptr, flags);
                auto indirectsDevice = glMapBufferRange(GL_DRAW_INDIRECT_BUFFER, 0, size, flags);
                auto indirectsHost = indirects.data() + cmdNext;
                memcpy(indirectsDevice, indirectsHost, size);
                glUnmapBuffer(GL_DRAW_INDIRECT_BUFFER);

                glMultiDrawElementsIndirect(
                    GL_TRIANGLES, eti.first, nullptr, currentBatchSize, 0);

                cmdNext += currentBatchSize;
                cmdRemain -= currentBatchSize;

                buffers.push_back(bufIndirect);
            }
        }

        glDeleteBuffers(buffers.size(), buffers.data());
    }
}

void
GL_Color_Pass::RenderSolidColor(
    Material_Instances const &cmds,
    glm::mat4 const &matVP) {

    for (auto &kv : cmds) {
        auto material = kv.first;

        auto *materialData
            = (topo::Material_Solid_Color *)_materialManager->GetMaterialData(
                material);

        gl::SetUniformLocation(
            _shaderSolidColor->locColor(),
            { materialData->color[0], materialData->color[1],
              materialData->color[2] });

        for (auto &cmd : kv.second) {
            auto &transform = cmd.transform;
            auto model = cmd.model;
            auto matTransform = glm::translate(transform.position)
                * glm::mat4_cast(transform.rotation)
                * glm::scale(transform.scale);

            gl::SetUniformLocation(
                _shaderSolidColor->locMatMVP(), matVP * matTransform);
            gl::SetUniformLocation(
                _shaderSolidColor->locMatModel(), matTransform);

            GLuint indexOffset;
            GLuint baseVertex;
            GLenum elementType;
            size_t numElements;
            GLuint firstIndex;

            _modelManager->GetDrawParameters(
                model, &indexOffset, &baseVertex, &elementType, &numElements, &firstIndex);

            glDrawElementsBaseVertex(
                GL_TRIANGLES, numElements, elementType, (void *)indexOffset,
                baseVertex);
        }
    }
}

void
Shader_Solid_Color::Build() {
    printf("[ topo ] building shader 'solid color'\n");
    auto vsh = FromStringLoadShader<GL_VERTEX_SHADER>(solid_color_vsh_glsl);
    auto fsh = FromStringLoadShader<GL_FRAGMENT_SHADER>(solid_color_fsh_glsl);

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

void
Shader_Lines::Build() {
    printf("[ topo ] building shader 'lines'\n");
    auto vsh = FromStringLoadShader<GL_VERTEX_SHADER>(lines2_vsh_glsl);
    auto fsh = FromStringLoadShader<GL_FRAGMENT_SHADER>(lines2_fsh_glsl);

    auto builder = gl::Shader_Program_Builder();
    auto program = builder.Attach(vsh).Attach(fsh).Link();
    if (program) {
        _program = std::move(program.value());

        _locMatMVP = gl::Uniform_Location<glm::mat4>(_program, "matMVP");
        _locColor = gl::Uniform_Location<glm::vec3>(_program, "color");
    } else {
        throw Shader_Linker_Exception("lines", builder.Error());
    }
}

void
Shader_Textured_Unlit::Build() {
    printf("[ topo ] building shader 'textured unlit'\n");
    auto vsh = FromStringLoadShader<GL_VERTEX_SHADER>(textured_unlit_vsh_glsl);
    auto fsh = FromStringLoadShader<GL_FRAGMENT_SHADER>(textured_unlit_fsh_glsl);

    auto builder = gl::Shader_Program_Builder();
    auto program = builder.Attach(vsh).Attach(fsh).Link();
    if (program) {
        _program = std::move(program.value());

        _locMatVP = gl::Uniform_Location<glm::mat4>(_program, "matVP");
        _locMatModel = gl::Uniform_Location<glm::mat4>(_program, "matModel");
        _locTexDiffuse = gl::Uniform_Location<GLint>(_program, "texDiffuse");
    } else {
        throw Shader_Linker_Exception("textured unlit", builder.Error());
    }
}

}
