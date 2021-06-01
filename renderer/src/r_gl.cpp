// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: OpenGL renderer
//

#include "r_renderer.h"
#include "r_queue.h"
#include <imgui_impl_opengl3.h>

#include <glm/vec3.hpp>
#include <glm/gtx/transform.hpp>

#define GLRES_GLM
#include <glm/gtc/type_ptr.hpp>
#include "glres.h"
#include <optional>
#include <fstream>
#include <sstream>
#include <queue>
#include <functional>
#include <array>
#include <list>

#include "r_gl_shadercompiler.h"
#include "shader_program_builder.h"
#include "shader_generic.h"

#include <Tracy.hpp>

using Mat3 = glm::mat3;
using Mat4 = glm::mat4;
using Vec3 = glm::vec3;
using Vec4 = glm::vec4;
using Quat = glm::quat;

extern "C" {
    extern char const* lines_vsh_glsl;
    extern char const* lines_fsh_glsl;
    extern char const* points_vsh_glsl;
    extern char const* points_fsh_glsl;
    extern char const* deferred_vsh_glsl;
    extern char const* deferred_fsh_glsl;
}

struct Texture {
    gl::Texture texture;
};

struct Model {
    gl::VAO vao;
    unsigned num_vertices;
    gl::VBO vertices;
    gl::VBO uvs;

    gl::VBO tangents;
    gl::VBO bitangents;
    gl::VBO normals;
};

static void GLMessageCallback
(GLenum src, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* lparam) {
    if (length == 0) return;
    if (severity == GL_DEBUG_SEVERITY_HIGH) {
        printf("[ gfx ] BACKEND ERROR!! '%s'\n", message);
    }
#ifndef NDEBUG
    else if (severity == GL_DEBUG_SEVERITY_MEDIUM) {
        printf("[ gfx ] BACKEND WARNING: '%s'\n", message);
    } else if (severity == GL_DEBUG_SEVERITY_LOW) {
        printf("[ gfx ] backend warning: '%s'\n", message);
    } else if (severity == GL_DEBUG_SEVERITY_NOTIFICATION) {
        printf("[ gfx ] backend note: '%s'\n", message);
    }
#endif
}

struct Line {
    gl::VAO arr;
    gl::VBO buf[2];
};

struct Point {
    gl::VAO arr;
    gl::VBO buf;
};

struct Element_Model {
    gl::VAO arr;
    gl::VBO vertices, elements, colors;
};

/**
 * VAO and VBO recycler for stream drawing.
 */
template<typename Tuple>
struct Array_Recycler {
public:
    /**
     * Get an unused instance of `Tuple`.
     * @param out Where the pointer to the tuple will be placed.
     * @return Handle to the instance.
     */
    size_t get(Tuple** out) {
        *out = NULL;
        if (ready_queue.empty()) {
            return make_new(out);
        } else {
            auto ret = ready_queue.front();
            ready_queue.pop();
            *out = &arrays[ret];
            return ret;
        }
    }

    /**
     * Mark a tuple instance as used and retire it.
     * @param handle An instance handle returned from get(Tuple**).
     */
    void put_back(size_t handle) {
        assert(handle < arrays.size());
        retired_queue.push(handle);
    }

    /**
     * Called after a frame ends.
     */
    void flip() {
        while (!retired_queue.empty()) {
            auto h = retired_queue.front();
            retired_queue.pop();
            ready_queue.push(h);
        }

        assert(retired_queue.size() == 0);
        assert(ready_queue.size() == arrays.size());
    }

    long long count() const {
        return retired_queue.size();
    }
protected:
    size_t make_new(Tuple** out) {
        auto ret = arrays.size();
        arrays.push_back(Tuple());
        *out = &arrays.back();
        return ret;
    }
private:
    std::queue<size_t> ready_queue;
    std::queue<size_t> retired_queue;
    std::vector<Tuple> arrays;
};

class GL_Renderer : public gfx::IRenderer {
public:
    GL_Renderer() {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        if (glDebugMessageCallback) {
            glDebugMessageCallback(GLMessageCallback, 0);
        } else {
            printf("[ gfx ] BACKEND WARNING: no messages will be received from the driver!\n");
        }

        ImGui_ImplOpenGL3_Init("#version 130");

        m_view = glm::translate(Vec3(0.0f, 0.0f, 0.0f));
        m_proj = glm::perspective(glm::radians(90.0f), 720.0f / 1280.0f, 0.01f, 8192.0f);

        std::optional<gl::Shader_Program> discard;
        LoadShaderFromStrings("Point cloud", points_vsh_glsl, points_fsh_glsl, {}, discard, [&](gl::Shader_Program &program) {
            auto const locView = gl::Uniform_Location<Mat4>(program, "matView");
            auto const locProj = gl::Uniform_Location<Mat4>(program, "matProj");
            auto const locModel = gl::Uniform_Location<Mat4>(program, "matModel");
            m_point_cloud_shader = { std::move(program), locView, locProj, locModel };
        });

        LoadShaderFromStrings("Lines", lines_vsh_glsl, lines_fsh_glsl, {}, discard, [&](gl::Shader_Program &program) {
            auto locMVP = gl::Uniform_Location<Mat4>(program, "matMVP");
            auto locColor0 = gl::Uniform_Location<Vec3>(program, "vColor0");
            auto locColor1 = gl::Uniform_Location<Vec3>(program, "vColor1");
            m_line_shader = { std::move(program), locMVP, locColor0, locColor1 };
        });
        
        m_element_model_shader = Shader_Generic();
        try_build(*m_element_model_shader);

        m_element_model_shader_with_vtx_color = Shader_Generic_With_Vertex_Colors();
        try_build(*m_element_model_shader_with_vtx_color);

        m_element_model_shader_textured = Shader_Generic_Textured_Unlit();
        try_build(*m_element_model_shader_textured);

        m_element_model_shader_textured_lit = Shader_Generic_Textured_Lit();
        try_build(*m_element_model_shader_textured_lit);

        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);
        glLineWidth(2.0f);
        glFrontFace(GL_CCW);

        // unsigned width, height;
        // get_resolution(&width, &height);
        // auto gbuf = G_Buffer::make_gbuffer(width, height);
        // assert(gbuf.has_value());
        // g_buffer = std::move(gbuf.value());
    }

    ~GL_Renderer() override {
        ImGui_ImplOpenGL3_Shutdown();
    }

    void try_build(Shader_Generic_Base &program) {
        try {
            program.build();
        } catch (Shader_Compiler_Exception const &ex) {
            printf("renderer: error while compiling shader program '%s':\n\tStage: ", program.name());
            switch (ex.stageKind()) {
            case GL_VERTEX_SHADER:
                printf("vertex");
                break;
            case GL_FRAGMENT_SHADER:
                printf("fragment");
                break;
            default:
                printf("unknown");
                break;
            }
            printf("\n\tWith defines:\n");
            for (auto &def : program.defines()) {
                printf("\t\t%s=%s\n", def.key.c_str(), def.value.c_str());
            }
            printf("\tMessage:\n%s\n", ex.errorMessage().c_str());
        } catch (Shader_Linker_Exception const &ex) {
            printf("renderer: error while linking shader program '%s':\n", program.name());
            printf("\tWith defines:\n");
            for (auto &def : program.defines()) {
                printf("\t\t%s=%s\n", def.key.c_str(), def.value.c_str());
            }
            printf("\tMessage:\n%s\n", ex.errorMessage().c_str());
        }
    }

    /**
     * Create a shader program from in-memory source code and execute a
     * callback on success.
     * 
     * \param name Name of the program (used for error messages)
     * \param srcVertex Vertex shader source code
     * \param srcFragment Fragment shader source code
     * \param defines A list of macros to define in the source code
     * \param out The shader program will be put here on success
     * \param cbOnLinkSuccess Callback to execute on success
     */
    void LoadShaderFromStrings(
        char const *name,
        char const *srcVertex,
        char const *srcFragment,
        Shader_Define_List const &defines,
        std::optional<gl::Shader_Program> &out,
        std::function<void(gl::Shader_Program &)> const &cbOnLinkSuccess) {

        try {
            auto vsh = FromStringLoadShader<GL_VERTEX_SHADER>(srcVertex, defines);
            auto fsh = FromStringLoadShader<GL_FRAGMENT_SHADER>(srcFragment, defines);

            auto builder = gl::Shader_Program_Builder(name);
            auto program = builder.Attach(vsh).Attach(fsh).Link();
            if (program) {
                cbOnLinkSuccess(program.value());
                out = std::move(program.value());
            } else {
                printf("Error linking shader '%s'\n", name);
                printf("\tWith defines:\n");
                for (auto &def : defines) {
                    printf("\t\t#define %s (%s)", def.key.c_str(), def.value.c_str());
                }

                printf("\tMessage:\n%s\n", builder.Error());
            }
        } catch (Shader_Compiler_Exception const &ex) {
            printf("Error compiling shader '%s':\n", name);
            switch (ex.stageKind()) {
            case GL_VERTEX_SHADER:
                printf("\tKind: Vertex\n");
                break;
            case GL_FRAGMENT_SHADER:
                printf("\tKind: Fragment\n");
                break;
            default:
                printf("\tKind: UNKNOWN\n");
                break;
            }
            printf("\tWith defines:\n");
            for (auto &def : defines) {
                printf("\t\t#define %s (%s)", def.key.c_str(), def.value.c_str());
            }
            printf("\tMessage:\n%s\n", ex.errorMessage().c_str());
        }
    }

    void set_camera(Mat4 const& view_matrix) override {
        m_view = view_matrix;
    }

    void get_camera(glm::mat4 &view_matrix, glm::mat4 &projection_matrix) override {
        view_matrix = m_view;
        projection_matrix = m_proj;
    }

    void draw_points(size_t nCount, Vec3 const* pPoints, Vec3 const& vWorldPosition) override {
        ZoneScoped;
        if (m_point_cloud_shader.has_value()) {
            Point* p;
            auto h_p = point_recycler.get(&p);
            glBindVertexArray(p->arr);

            auto const size = nCount * 3 * sizeof(float);
            auto const data = pPoints;
            glBindBuffer(GL_ARRAY_BUFFER, p->buf);
            glBufferData(GL_ARRAY_BUFFER, size, data, GL_STREAM_DRAW);

            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
            glEnableVertexAttribArray(0);

            auto& shader = m_point_cloud_shader.value();
            glUseProgram(shader.program);
            
            auto const matModel = glm::translate(vWorldPosition);

            gl::SetUniformLocation(shader.locModelMatrix , matModel);
            gl::SetUniformLocation(shader.locMatView, m_view);
            gl::SetUniformLocation(shader.locMatProj, m_proj);

            glDrawArrays(GL_POINTS, 0, nCount);

            point_recycler.put_back(h_p);
        } else {
            printf("Can't draw: no shader!\n");
        }
    }

    void push_debug_group(char const* pszFormat, ...) {
        char buffer[256];
        va_list args;
        va_start(args, pszFormat);
        auto len = vsnprintf(buffer, 255, pszFormat, args);
        buffer[255] = 0;
        glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, len, buffer);
        va_end(args);
    }

    void pop_debug_group() {
        glPopDebugGroup();
    }

    void new_frame() override {
        FrameMark;
        ZoneScoped;
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        ImGui_ImplOpenGL3_NewFrame();

        line_recycler.flip();
        point_recycler.flip();
        element_model_recycler.flip();
    }

    double present() override {
        TracyPlot("GL::line_recycler::count", line_recycler.count());
        TracyPlot("GL::point_recycler::count", point_recycler.count());
        TracyPlot("GL::element_model_recycler::count", element_model_recycler.count());

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        return 0;
    }

    void change_resolution(unsigned* inout_width, unsigned* inout_height) override {
        m_proj = glm::perspective(glm::radians(90.0f), (*inout_width) / (float)(*inout_height), 0.01f, 8192.0f);
        surf_width = *inout_width;
        surf_height = *inout_height;

        // g_buffer = std::move(G_Buffer::make_gbuffer(surf_width, surf_height).value());
    }

    void get_resolution(unsigned* out_width, unsigned* out_height) override {
        *out_width = surf_width;
        *out_height = surf_height;
    }

    void draw_lines(
        Vec3 const* pEndpoints,
        size_t nLineCount,
        Vec3 const& vWorldPosition,
        Vec3 const& vStartColor,
        Vec3 const& vEndColor
    ) override {
        ZoneScoped;
        if (nLineCount == 0) return;

        Line* l;
        auto l_h = line_recycler.get(&l);

        glBindVertexArray(l->arr);

        auto zero_one = std::make_unique<float[]>(nLineCount * 2);
        auto zero_one_p = zero_one.get();
        for (unsigned i = 0; i < nLineCount * 2; i++) {
            zero_one_p[i] = ((i & 1) == 0) ? 0.0f : 1.0f;
        }

        auto const size = nLineCount * 2 * 3 * sizeof(float);
        auto const data = pEndpoints;

        glBindBuffer(GL_ARRAY_BUFFER, l->buf[0]);
        glBufferData(GL_ARRAY_BUFFER, size, data, GL_STREAM_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);

        glBindBuffer(GL_ARRAY_BUFFER, l->buf[1]);
        glBufferData(GL_ARRAY_BUFFER, nLineCount * 2 * sizeof(float), zero_one_p, GL_STREAM_DRAW);
        glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 1 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(1);

        if (m_line_shader.has_value()) {
            auto& shader = m_line_shader.value();
            glUseProgram(shader.program);
            auto matModel = glm::translate(vWorldPosition);
            auto matMVP = m_proj * m_view * matModel;
            gl::SetUniformLocation(shader.locMVP, matMVP);
            gl::SetUniformLocation(shader.locColor0, vStartColor);
            gl::SetUniformLocation(shader.locColor1, vEndColor);

            glDrawArrays(GL_LINES, 0, 2 * nLineCount);
        } else {
            printf("Can't draw: no shader!\n");
        }
        line_recycler.put_back(l_h);
    }

    void draw_triangle_elements(
        size_t vertex_count,
        std::array<float, 3> const* vertices,
        size_t element_count,
        unsigned const* elements,
        glm::vec3 const& vWorldPosition
    ) override {
        ZoneScoped;
        if (element_count == 0) return;

        if (m_element_model_shader.has_value()) {
            Element_Model* mdl;
            auto mdl_h = element_model_recycler.get(&mdl);

            glBindVertexArray(mdl->arr);

            glBindBuffer(GL_ARRAY_BUFFER, mdl->vertices);
            glBufferData(GL_ARRAY_BUFFER, vertex_count * sizeof(glm::vec3), vertices, GL_STREAM_DRAW);

            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mdl->elements);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, element_count * sizeof(unsigned), elements, GL_STREAM_DRAW);

            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), nullptr);
            glEnableVertexAttribArray(0);

            auto& shader = *m_element_model_shader;
            glUseProgram(shader.program());
            auto matModel = glm::translate(vWorldPosition);
            auto matMVP = m_proj * m_view * matModel;
            gl::SetUniformLocation(shader.locMVP(), matMVP);

            glDrawElements(GL_TRIANGLES, element_count, GL_UNSIGNED_INT, 0);

            element_model_recycler.put_back(mdl_h);
        } else {
            printf("renderer: can't draw triangle elements: no shader!\n");
        }
    }

    void draw_triangle_elements_with_vertex_color(size_t vertex_count, std::array<float, 3> const* vertices, glm::u8vec3 const* vertex_colors, size_t element_count, unsigned const* elements, glm::vec3 const& vWorldPosition) override {
        ZoneScoped;
        if (element_count == 0) return;

        if (m_element_model_shader_with_vtx_color.has_value()) {
            Element_Model* mdl;
            auto mdl_h = element_model_recycler.get(&mdl);

            glBindVertexArray(mdl->arr);

            glBindBuffer(GL_ARRAY_BUFFER, mdl->vertices);
            glBufferData(GL_ARRAY_BUFFER, vertex_count * sizeof(glm::vec3), vertices, GL_STREAM_DRAW);

            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), nullptr);
            glEnableVertexAttribArray(0);

            glBindBuffer(GL_ARRAY_BUFFER, mdl->colors);
            glBufferData(GL_ARRAY_BUFFER, vertex_count * sizeof(glm::u8vec3), vertex_colors, GL_STREAM_DRAW);

            glVertexAttribPointer(1, 3, GL_UNSIGNED_BYTE, GL_FALSE, 3, nullptr);
            glEnableVertexAttribArray(1);

            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mdl->elements);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, element_count * sizeof(unsigned), elements, GL_STREAM_DRAW);

            auto& shader = *m_element_model_shader_with_vtx_color;
            glUseProgram(shader.program());
            auto matModel = glm::translate(vWorldPosition);
            auto matMVP = m_proj * m_view * matModel;
            gl::SetUniformLocation(shader.locMVP(), matMVP);

            glDrawElements(GL_TRIANGLES, element_count, GL_UNSIGNED_INT, 0);

            element_model_recycler.put_back(mdl_h);
        } else {
            printf("renderer: can't draw triangle elements with vtx color: no shader!\n");
        }
    }

    bool upload_texture(gfx::Texture_ID *out_id, unsigned width, unsigned height, gfx::Texture_Format format, void const *image) override {
        // Ptr to image data may be NULL only if the image is empty.
        if (image == nullptr && (width != 0 || height != 0)) {
            return false;
        }

        if (out_id == nullptr) {
            return false;
        }

        gl::Texture texture;
        glBindTexture(GL_TEXTURE_2D, texture);

        // Set up wrapping behavior
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        // Set up filtering
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        // Set up mipmap generation
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        switch (format) {
        case gfx::Texture_Format::RGB888:
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, image);
            break;
        }

        glGenerateMipmap(GL_TEXTURE_2D);

        _textures.push_back({ std::move(texture) });
        *out_id = &_textures.back();

        return true;
    }

    void destroy_texture(gfx::Texture_ID id) override {
        if (id == nullptr) {
            return;
        }

        std::remove_if(_textures.begin(), _textures.end(), [&](Texture const &t) { return &t == id; });
    }

    static glm::vec3 vec3_cast(std::array<float, 3> const &arr) {
        return glm::vec3(arr[0], arr[1], arr[2]);
    }

    static glm::vec2 vec2_cast(std::array<float, 2> const &arr) {
        return glm::vec2(arr[0], arr[1]);
    }

    bool create_model(gfx::Model_ID *out_id, gfx::Model_Descriptor const *model) override {
        if (out_id == nullptr || model == nullptr) {
            return false;
        }

        if (model->elements == nullptr || model->vertices == nullptr) {
            return false;
        }

        Model mdl;
        glBindVertexArray(mdl.vao);

        // We need to turn the indexed mesh into a basic one

        auto num_vertices = model->element_count;

        // Copy vertices and normals
        std::vector<std::array<float, 3>> vertices_buf;
        std::vector<std::array<float, 3>> normals_buf;
        vertices_buf.resize(num_vertices);
        normals_buf.resize(num_vertices);
        for (size_t i = 0; i < num_vertices; i++) {
            auto idx = model->elements[i];
            vertices_buf[i] = model->vertices[idx];
            normals_buf[i] = model->normals[idx];
        }

        if (model->vertices != nullptr && model->uv != nullptr && model->normals != nullptr) {
            // Precompute tangents and bitangents
            std::vector<glm::vec3> tangents(num_vertices);
            std::vector<glm::vec3> bitangents(num_vertices);
            auto num_triangles = model->element_count / 3;
            for (size_t t = 0; t < num_triangles; t++) {
                auto idx0 = model->elements[t * 3 + 0];
                auto idx1 = model->elements[t * 3 + 1];
                auto idx2 = model->elements[t * 3 + 2];

                auto p0 = vec3_cast(model->vertices[idx0]);
                auto p1 = vec3_cast(model->vertices[idx1]);
                auto p2 = vec3_cast(model->vertices[idx2]);
                auto w0 = vec2_cast(model->uv[t * 3 + 0]);
                auto w1 = vec2_cast(model->uv[t * 3 + 1]);
                auto w2 = vec2_cast(model->uv[t * 3 + 2]);

                auto e1 = p1 - p0;
                auto e2 = p2 - p0;
                auto x1 = w1.x - w0.x;
                auto x2 = w2.x - w0.x;
                auto y1 = w1.y - w0.y;
                auto y2 = w2.y - w0.y;

                auto r = 1.0f / (x1 * y2 - x2 * y1);
                assert(std::isfinite(r) && !std::isnan(r));
                auto tangent = normalize((e1 * y2 - e2 * y1) * r);
                auto bitangent = normalize((e2 * x1 - e1 * x2) * r);

                tangents[t * 3 + 0] = tangent;
                tangents[t * 3 + 1] = tangent;
                tangents[t * 3 + 2] = tangent;
                bitangents[t * 3 + 0] = bitangent;
                bitangents[t * 3 + 1] = bitangent;
                bitangents[t * 3 + 2] = bitangent;
            }

            glBindBuffer(GL_ARRAY_BUFFER, mdl.normals);
            glBufferData(GL_ARRAY_BUFFER, num_vertices * sizeof(model->normals[0]), normals_buf.data(), GL_STATIC_DRAW);
            glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), nullptr);
            glEnableVertexAttribArray(2);

            glBindBuffer(GL_ARRAY_BUFFER, mdl.tangents);
            glBufferData(GL_ARRAY_BUFFER, num_vertices * sizeof(glm::vec3), tangents.data(), GL_STATIC_DRAW);
            glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), nullptr);
            glEnableVertexAttribArray(3);

            glBindBuffer(GL_ARRAY_BUFFER, mdl.bitangents);
            glBufferData(GL_ARRAY_BUFFER, num_vertices * sizeof(glm::vec3), bitangents.data(), GL_STATIC_DRAW);
            glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), nullptr);
            glEnableVertexAttribArray(4);
        }

        glBindBuffer(GL_ARRAY_BUFFER, mdl.vertices);
        glBufferData(GL_ARRAY_BUFFER, num_vertices * sizeof(glm::vec3), vertices_buf.data(), GL_STATIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), nullptr);
        glEnableVertexAttribArray(0);
        vertices_buf.clear();

        // UVs are not indexed
        glBindBuffer(GL_ARRAY_BUFFER, mdl.uvs);
        glBufferData(GL_ARRAY_BUFFER, num_vertices * sizeof(glm::vec2), model->uv, GL_STATIC_DRAW);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), nullptr);
        glEnableVertexAttribArray(1);

        mdl.num_vertices = num_vertices;

        _models.push_back(std::move(mdl));
        *out_id = &_models.back();

        return true;
    }

    void destroy_model(gfx::Model_ID id) override {
        if (id == nullptr) {
            return;
        }

        std::remove_if(_models.begin(), _models.end(), [&](Model const &m) { return &m == id; });
    }

    void draw_textured_triangle_elements(
        gfx::Model_ID model_handle,
        gfx::Material_Unlit const &material,
        gfx::Transform const &transform
    ) override {
        if (model_handle == nullptr) {
            return;
        }

        if (m_element_model_shader_textured.has_value()) {
            auto model = (Model *)model_handle;
            glBindVertexArray(model->vao);

            auto& shader = *m_element_model_shader_textured;
            glUseProgram(shader.program());

            auto matTransform =
                glm::translate(transform.position) *
                glm::mat4_cast(transform.rotation) *
                glm::scale(transform.scale);

            auto matMVP = m_proj * m_view * matTransform;
            gl::SetUniformLocation(shader.locMVP(), matMVP);

            gl::SetUniformLocation(shader.locTintColor(), { 1, 1, 1, 1 });

            auto texDiffuse = (Texture *)material.diffuse;
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, texDiffuse->texture);
            gl::SetUniformLocation(shader.locTexDiffuse(), 0);

            glDrawArrays(GL_TRIANGLES, 0, model->num_vertices);
        } else {
            printf("renderer: can't draw textured triangle elements: no shader!\n");
        }
    }

    void draw_triangle_elements(
        gfx::Model_ID model_handle,
        gfx::Transform const &transform
    ) override {
        gfx::Render_Parameters rp;
        draw_triangle_elements(rp, model_handle, transform);
    }

    void draw_triangle_elements(
        gfx::Render_Parameters const &params,
        gfx::Model_ID model_handle,
        gfx::Transform const &transform
    ) override {
        if (model_handle == nullptr) {
            return;
        }

        if (m_element_model_shader.has_value()) {
            auto model = (Model *)model_handle;
            glBindVertexArray(model->vao);

            auto& shader = *m_element_model_shader;
            glUseProgram(shader.program());

            auto matTransform =
                glm::translate(transform.position) *
                glm::mat4_cast(transform.rotation) *
                glm::scale(transform.scale);

            auto matMVP = m_proj * m_view * matTransform;
            gl::SetUniformLocation(shader.locMVP(), matMVP);

            if (params.tint_color.has_value()) {
                gl::SetUniformLocation(shader.locTintColor(), params.tint_color.value());
            } else {
                gl::SetUniformLocation(shader.locTintColor(), { 1, 1, 1, 1 });
            }

            glDrawArrays(GL_TRIANGLES, 0, model->num_vertices);
        } else {
            printf("renderer: can't draw triangle elements: no shader!\n");
        }
    }

    virtual void draw_textured_triangle_elements(
        gfx::Model_ID model_handle,
        gfx::Material_Lit const &material,
        gfx::Transform const &transform) override {
        if (model_handle == nullptr) {
            return;
        }

        if (m_element_model_shader_textured_lit.has_value()) {
            auto model = (Model *)model_handle;
            glBindVertexArray(model->vao);

            auto& shader = *m_element_model_shader_textured_lit;
            glUseProgram(shader.program());

            auto matModel = glm::mat4_cast(transform.rotation) * glm::scale(transform.scale);
            auto matTransform = glm::translate(transform.position) * matModel;

            auto matMVP = m_proj * m_view * matTransform;
            gl::SetUniformLocation(shader.locMVP(), matMVP);

            auto matNormal =
                scale(1.0f / transform.scale) *
                mat4_cast(inverse(transform.rotation)) *
                translate(-transform.position);

            gl::SetUniformLocation(shader.locModelMatrix(), glm::mat4(matModel));

            gl::SetUniformLocation(shader.locTintColor(), { 1, 1, 1, 1 });

            gl::SetUniformLocation(shader.locSunPosition(), _sun_position);

            auto viewPos = glm::vec3(m_view[3]);
            gl::SetUniformLocation(shader.locViewPosition(), viewPos);

            auto texDiffuse = (Texture *)material.diffuse;
            glActiveTexture(GL_TEXTURE0 + 0);
            glBindTexture(GL_TEXTURE_2D, texDiffuse->texture);
            gl::SetUniformLocation(shader.locTexDiffuse(), 0);

            auto texNormal = (Texture *)material.normal;
            glActiveTexture(GL_TEXTURE0 + 1);
            glBindTexture(GL_TEXTURE_2D, texNormal->texture);
            gl::SetUniformLocation(shader.locTexNormal(), 1);

            glDrawArrays(GL_TRIANGLES, 0, model->num_vertices);
        } else {
            printf("renderer: can't draw textured triangle elements: no shader!\n");
        }
    }

    void set_sun_position(glm::vec3 const &position) override {
        _sun_position = position;
    }

private:
    Mat4 m_proj, m_view;
    unsigned surf_width = 256, surf_height = 256;

    Array_Recycler<Line> line_recycler;
    Array_Recycler<Point> point_recycler;
    Array_Recycler<Element_Model> element_model_recycler;

    glm::vec3 _sun_position = { 0, 1000, 0 };

    struct Line_Shader {
        gl::Shader_Program program;
        gl::Uniform_Location<Mat4> locMVP;
        gl::Uniform_Location<Vec3> locColor0, locColor1;
    };

    struct Point_Cloud_Shader {
        gl::Shader_Program program;
        gl::Uniform_Location<Mat4> locMatView, locMatProj, locModelMatrix;
    };

    std::optional<Line_Shader> m_line_shader;
    std::optional<Point_Cloud_Shader> m_point_cloud_shader;
    std::optional<Shader_Generic> m_element_model_shader;
    std::optional<Shader_Generic_With_Vertex_Colors> m_element_model_shader_with_vtx_color;
    std::optional<Shader_Generic_Textured_Unlit> m_element_model_shader_textured;
    std::optional<Shader_Generic_Textured_Lit> m_element_model_shader_textured_lit;

    std::list<Texture> _textures;
    std::list<Model> _models;
};

std::unique_ptr<gfx::IRenderer> gfx::make_opengl_renderer(void* glctx, void* (*getProcAddress)(char const*)) {
    if (gladLoadGLLoader(getProcAddress) == 1) {
        return std::make_unique<GL_Renderer>();
    } else {
        return nullptr;
    }
}
