// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: 
//

#include <list>

#include <glm/mat3x3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <topo.h>

#include "gl_color_pass.h"
#include "gl_depth_prepass.h"
#include "gl_gbuffer.h"
#include "gl_model_manager.h"
#include "gl_shadercompiler.h"
#include "gl_texture_manager.h"
#include "material_manager.h"
#include "render_queue.h"
#include "renderable_manager.h"
#include "shader_generic.h"

#include <glad/glad.h>
#include <imgui_impl_opengl3.h>
#include <trigen/mesh_compress.h>

#include <Tracy.hpp>

namespace topo {

static void GLMessageCallback
(GLenum src, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* lparam) {
    if (length == 0)
        return;
    if (severity == GL_DEBUG_SEVERITY_HIGH) {
        printf("[ topo ] ERROR: %s\n", message);
    } else if (severity == GL_DEBUG_SEVERITY_MEDIUM) {
        printf("[ topo ] WARNING: %s\n", message);
    } else if (severity == GL_DEBUG_SEVERITY_LOW) {
        printf("[ topo ] %s\n", message);
    } else if (severity == GL_DEBUG_SEVERITY_NOTIFICATION) {
#if 0
        printf("[ topo ] %s\n", message);
#endif
    }
}

class Instance : public IInstance {
public:
    Instance(void *glctx, void *imguiContext)
        : _glCtx(glctx)
        , _imguiCtx(imguiContext)
        , _colorPass(
              &_modelManager,
              &_renderableManager,
              &_materialManager,
              &_textureManager,
              &_shaderTexturedUnlit,
              &_shaderSolidColor,
              &_shaderLines) {
        ImGui::SetCurrentContext(ImguiContext());
        ImGui_ImplOpenGL3_Init("#version 130");

        if (glDebugMessageCallback) {
            glDebugMessageControl(
                GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, NULL, GL_TRUE);
            glDebugMessageCallback(GLMessageCallback, 0);
        } else {
            printf("[ topo ] BACKEND WARNING: no messages will be received from the driver!\n");
        }

        _shaderTexturedUnlit.Build();
        _shaderDepthPass.build();
        _shaderSolidColor.Build();
        _shaderLines.Build();

        _matView = glm::mat4(1.0f);
        _matProj = glm::perspective(glm::radians(90.0f), 1.0f, 0.1f, 10000.0f);

        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LEQUAL);
        glLineWidth(2.0f);
        glFrontFace(GL_CCW);

        glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &_prevFbDraw);
        glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &_prevFbRead);
    }

    ~Instance() override {
        ImGui::SetCurrentContext(ImguiContext());
        ImGui_ImplOpenGL3_Shutdown();
    }

    void *
    GLContext() override {
        return _glCtx;
    }

    void
    NewFrame() override {
        FrameMark;
        ImGui::SetCurrentContext(ImguiContext());
        ImGui_ImplOpenGL3_NewFrame();
    }

    void
    Present() override {
        ImGui::SetCurrentContext(ImguiContext());

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    }

    void
    ResolutionChanged(unsigned width, unsigned height) override {
        _matProj = glm::perspective(glm::radians(90.0f), float(height) / float(width), 0.1f, 1000.0f);
        RecalculateVP();
        glViewport(0, 0, width, height);

        _gbuffer = G_Buffer("main", width, height);
        _depthPrepass = GL_Depth_Prepass(
            &_modelManager, &_renderableManager, width, height,
            &_shaderDepthPass);

        _width = width;
        _height = height;
    }

    ImGuiContext *
    ImguiContext() override {
        return (ImGuiContext *)_imguiCtx;
    }

    bool
    CreateTexture(
        Texture_ID *outHandle,
        unsigned width,
        unsigned height,
        Texture_Format format,
        void const *image) override {
        return _textureManager.CreateTexture(
            outHandle, width, height, format, image);
    }

    void
    DestroyTexture(Texture_ID texture) override {
        _textureManager.DestroyTexture(texture);
    }

    void
    BeginModelManagement() override {
        _modelManagementInProgress = true;
    }

    void
    FinishModelManagement() override {
        _modelManagementInProgress = false;
        _modelManager.Regenerate();
    }

    bool
    CreateModel(Model_ID *outHandle, Model_Descriptor const *descriptor)
        override {
        auto ret = _modelManager.CreateModel(outHandle, descriptor);

        if (!_modelManagementInProgress) {
            _modelManager.Regenerate();
        }

        return ret;
    }

    void
    DestroyModel(Model_ID model) override {
        _modelManager.DestroyModel(model);

        if (!_modelManagementInProgress) {
            _modelManager.Regenerate();
        }
    }

    bool
    CreateUnlitMaterial(Material_ID *outHandle, Texture_ID diffuse) override {
        return _materialManager.CreateUnlitMaterial(outHandle, diffuse);
    }

    bool
    CreateUnlitTransparentMaterial(Material_ID *outHandle, Texture_ID diffuse)
        override {
        return _materialManager.CreateUnlitTransparentMaterial(
            outHandle, diffuse);
    }

    bool
    CreateLitMaterial(
        Material_ID *outHandle,
        Texture_ID diffuse,
        Texture_ID normal) override {
        return _materialManager.CreateLitMaterial(outHandle, diffuse, normal);
    }

    bool
    CreateSolidColorMaterial(Material_ID *outHandle, glm::vec3 const &color)
        override {
        return _materialManager.CreateSolidColorMaterial(outHandle, color);
    }

    void
    DestroyMaterial(Material_ID material) override {
        _materialManager.DestroyMaterial(material);
    }

    bool
    CreateRenderable(
        Renderable_ID *outHandle,
        Model_ID model,
        Material_ID material) override {
        return _renderableManager.CreateRenderable(outHandle, model, material);
    }

    void
    DestroyRenderable(Renderable_ID renderable) override {
        _renderableManager.DestroyRenderable(renderable);
    }

    IRender_Queue *
    BeginRendering() override {
        if (_renderQueue == nullptr) {
            _renderQueue = std::make_unique<Render_Queue>();
        } else {
            fprintf(
                stderr,
                "[ topo ] Warning: BeginRendering was called multiple times\n");
        }

        return _renderQueue.get();
    }

    void
    FinishRendering() override {
        ZoneScoped;
        if (!_renderQueue) {
            fprintf(
                stderr,
                "[ topo ] Warning: FinishRendering was without a matching call "
                "to BeginRendering!\n");
            return;
        }

        _modelManager.BindMegabuffer();

        static bool doDepthPrepass = true;

        if (ImGui::Begin("Renderer")) {
            ImGui::Checkbox("Enable depth prepass", &doDepthPrepass);
        }
        ImGui::End();

        auto multiDraw = GL_Multidraw(
            _renderQueue.get(), &_renderableManager, &_materialManager,
            &_modelManager);

        if (doDepthPrepass) {
            _depthPrepass->Execute(_renderQueue.get(), multiDraw, _matVP);
        }

        _gbuffer->activate();
        // Copy result of the depth prepass into the G-buffer

        if (doDepthPrepass) {
            _depthPrepass->BlitDepth(_gbuffer->GetFramebuffer());

            glDepthFunc(GL_LEQUAL);
            glDepthMask(GL_FALSE);
        }

        glClear(GL_COLOR_BUFFER_BIT);

        _colorPass.Execute(_renderQueue.get(), multiDraw, _matVP);

        _gbuffer->draw(_renderQueue.get(), _matView[3], _prevFbRead, _prevFbDraw, _width, _height);

        _renderQueue.reset();
    }

    void
    SetEyeCamera(Transform const &transform) override {
        auto matRotation = glm::mat4_cast(transform.rotation);
        auto matTranslation = glm::translate(matRotation, transform.position);
        _matView = matTranslation;
        RecalculateVP();
    }

    void
    SetEyeViewMatrix(glm::mat4 const &matView) override {
        _matView = matView;
        RecalculateVP();
    }

    bool
    CreateRenderableLinesStreaming(
        Renderable_ID *outHandle,
        glm::vec3 const *endpoints,
        size_t lineCount,
        glm::vec3 const &colorBegin,
        glm::vec3 const &colorEnd) override {
        return _renderableManager.CreateRenderableLinesStreaming(
            outHandle, endpoints, lineCount, colorBegin, colorEnd);
    }

    void
    RecalculateVP() {
        _matVP = _matProj * _matView;
    }

private:
    void *_glCtx;
    void *_imguiCtx;

    UPtr<Render_Queue> _renderQueue;

    GL_Model_Manager _modelManager;
    GL_Texture_Manager _textureManager;
    Material_Manager _materialManager;
    Renderable_Manager _renderableManager;

    Shader_Textured_Unlit _shaderTexturedUnlit;
    Shader_Solid_Color _shaderSolidColor;
    Shader_Lines _shaderLines;

    std::optional<G_Buffer> _gbuffer;

    GL_Depth_Pass_Shader _shaderDepthPass;
    std::optional<GL_Depth_Prepass> _depthPrepass;

    GL_Color_Pass _colorPass;

    glm::mat4 _matProj;
    glm::mat4 _matView;

    glm::mat4 _matVP;

    bool _modelManagementInProgress = false;

    unsigned _width, _height;
    GLint _prevFbDraw, _prevFbRead;
};

UPtr<IInstance>
MakeInstance(
    void *glctx,
    void *(*getProcAddress)(char const *),
    void *imguiContext) {
    if (!glctx || !getProcAddress) {
        return nullptr;
    }

    if (!gladLoadGLLoader(getProcAddress)) {
        fprintf(stderr, "GLAD failed\n");
        return nullptr;
    }

    return std::make_unique<Instance>(glctx, imguiContext);
}

}
