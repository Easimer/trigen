// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: Generate 3D mesh from a Lindenmayer-system
//

#include <cassert>
#include <ctime>
#include <optional>
#include <fstream>
#include <sstream>
#include <stack>
#include <SDL.h>
#include "glad/glad.h"
#include <trigen/sdl_helper.h>
#include <trigen/linear_math.h>
#include "general.h"
#include "glres.h"
#include "meshbuilder.h"
#include "trunk_generator.h"

struct GL_Renderer : public sdl::Renderer {
    SDL_GLContext glctx;

    GL_Renderer() : Renderer("cylindergen", 1280, 720, SDL_WINDOW_SHOWN | SDL_WINDOW_OPENGL) {
        if (window != NULL && renderer != NULL) {
            SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
            SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 5);
            SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
            SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
            SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
            SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 1);
            SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 4);

            glctx = SDL_GL_CreateContext(window);
        }
    }

    ~GL_Renderer() {
        if (glctx != NULL) {
            SDL_GL_DeleteContext(glctx);
        }
    }

    operator bool() const {
        return window && renderer && glctx;
    }

    void Present() const {
        SDL_GL_SwapWindow(window);
    }
};

static void GLMessageCallback
(GLenum src, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* lparam) {
    if (length == 0) return;
    if (severity == GL_DEBUG_SEVERITY_HIGH) {
        printf("[ gfx ] BACKEND ERROR!! '%s'\n", message);
        assert(0);
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

void Bind(gl::VBO const& vbo) {
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
}

void BindElements(gl::VBO const& vbo) {
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo);
}

void Bind(gl::VAO const& vao) {
    glBindVertexArray(vao);
}

static void UploadStaticVertices(gl::VBO const& vbo, GLsizei size, void const* data) {
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, size, data, GL_STATIC_DRAW);
}

static void UploadElementArray(gl::VBO const& vbo, GLsizei size, void const* data) {
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, size, data, GL_STATIC_DRAW);
}

template<typename Shader>
static bool CompileShaderFromString(Shader const& shader, char const* pszSource) {
    GLint bSuccess;
    char const* aSources[1] = { pszSource };
    glShaderSource(shader, 1, aSources, NULL);
    glCompileShader(shader);
    glGetShaderiv(shader, GL_COMPILE_STATUS, &bSuccess);

    if (bSuccess == 0) {
        char pchMsgBuf[128];
        glGetShaderInfoLog(shader, 128, NULL, pchMsgBuf);
        printf("CompileShaderFromString failed: %s\n", pchMsgBuf);
    }

    return bSuccess != 0;
}

static gl::VAO BuildVAO(gl::VBO const& hPos) {
    gl::VAO vao;

    glBindVertexArray(vao);
    // Position buffer
    glBindBuffer(GL_ARRAY_BUFFER, hPos);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    return vao;
}

struct Element_Model {
    gl::VAO vao;
    gl::VBO vbo_vertices, vbo_elements;
    const size_t elements;
};

struct Draw_Load_Unit {
    gl::Shader_Program program;
    Element_Model mdl;
};

struct Camera_State {
    lm::Vector4 position;
    lm::Vector4 euler_rotation;
    lm::Matrix4 mvp;
};

static Element_Model BuildModel(Mesh_Builder::Optimized_Mesh const& opt) {
    gl::VAO vao;
    gl::VBO vbo_vertices;
    gl::VBO vbo_elements;

    Bind(vao);
    UploadStaticVertices(vbo_vertices, opt.VerticesSize(), opt.vertices.data());
    UploadElementArray(vbo_elements, opt.ElementsSize(), opt.elements.data());

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    return {
        std::move(vao),
        std::move(vbo_vertices), std::move(vbo_elements),
        opt.elements.size() };
}

static void Draw(Element_Model const& elemmdl) {
    Bind(elemmdl.vao);
    glDrawElements(GL_TRIANGLES, elemmdl.elements, GL_UNSIGNED_INT, 0);
}

static void RenderLoop(GL_Renderer& r, Draw_Load_Unit const& dlu) {
    bool bExit = false;
    Camera_State cam;
    lm::Matrix4 matProj, matInvProj;
    lm::Perspective(matProj, matInvProj, r.width, r.height, 1.57079633f, 0.01f, 1000.0f);
    cam.position = lm::Vector4(0, 32, 128);

    gl::Uniform_Location<lm::Matrix4> locMVP(dlu.program, "matMVP");

    float flZoom = 1.0f;

    while (!bExit) {
        SDL_Event ev;
        while (SDL_PollEvent(&ev)) {
            switch (ev.type) {
                case SDL_QUIT: {
                    bExit = true;
                    break;
                }
                case SDL_MOUSEMOTION: {
                    if (ev.motion.state & SDL_BUTTON_RMASK) {
                        cam.euler_rotation = cam.euler_rotation + lm::Vector4(0, ev.motion.xrel / 32.0f, 0);
                    }
                    break;
                }
                case SDL_MOUSEWHEEL: {
                    flZoom += -ev.wheel.y;
                    break;
                }
            }
        }

        if (flZoom < 1.0f) { flZoom = 1.0f; }

        // Arcball camera
        auto matView = lm::Scale(1.0f / flZoom) * lm::RotationY(cam.euler_rotation[1]) * lm::Translation(-cam.position[0], -cam.position[1], -cam.position[2]);
        auto matMVP = matView * matProj;

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glUseProgram(dlu.program);
        gl::SetUniformLocation(locMVP, matMVP);
        Draw(dlu.mdl);

        r.Present();
    }
}

template<GLenum kType>
static std::optional<gl::Shader<kType>> FromFileLoadShader(char const* pszPath) {
    gl::Shader<kType> shader;

    std::ifstream f(pszPath);
    if (f) {
        std::stringstream ss;
        ss << f.rdbuf();
        if (CompileShaderFromString(shader, ss.str().c_str())) {
            return shader;
        }
    }


    return {};
}

static float randf() {
    return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

template<size_t N, typename... T>
struct Rule_Set_Impl {
    constexpr std::string const& F(char const) const {
        throw std::exception();
        return empty;
    }

    std::string const empty;
};

enum class Lindenmayer_Op {
    Noop,
    Forward,
    Push, Pop,
    Yaw_Pos, Yaw_Neg,
    Pitch_Pos, Pitch_Neg,
    Roll_Pos, Roll_Neg,
};

class Lindenmayer_System {
public:
    using Alphabet = std::unordered_map<char, std::vector<Lindenmayer_Op>>;
    using Rule_Set = std::unordered_map<char, std::string>;

    Lindenmayer_System(std::string const& axiom, Alphabet const& alphabet, Rule_Set const& rules)
    : axiom(axiom), alphabet(alphabet), rules(rules) {
        // Check that every character is present in the alphabet
        for (auto& ch : axiom) {
            assert(alphabet.count(ch));
        }
        for (auto& rule : rules) {
            assert(alphabet.count(rule.first) > 0);
            for (auto& ch : rule.second) {
                assert(alphabet.count(ch) > 0);
            }
        }
    }

    std::vector<Lindenmayer_Op> Iterate(size_t unIterations) const {
        std::vector<Lindenmayer_Op> ret;
        std::string current = axiom;
        std::string buf;

        for (size_t uiCurrentIteration = 0; uiCurrentIteration < unIterations; uiCurrentIteration++) {
            for (size_t iOff = 0; iOff < current.size(); iOff++) {
                auto cur = current[iOff];
                assert(alphabet.count(cur) == 1);
                if (rules.count(cur)) {
                    buf += rules.at(cur);
                } else {
                    buf += cur;
                }
            }
            current = std::move(buf);
        }

        ret.reserve(current.size());
        for (auto ch : current) {
            auto const& Y = alphabet.at(ch);
            for (auto op : Y) {
                ret.push_back(op);
            }
        }

        return ret;
    }
private:
    std::string const axiom;
    Alphabet const alphabet;
    Rule_Set const rules;
};

struct Execution_State {
    uint32_t uiNodeCurrent;
    float flYaw, flPitch, flRoll;
};

lm::Vector4 GetDirectionVector(Execution_State const& s) {
    auto u = cosf(s.flYaw) * cosf(s.flPitch);
    auto v = cosf(s.flPitch) * sinf(s.flYaw);
    auto w = sinf(s.flPitch);
    return lm::Vector4(-v, w, -u);
}

Tree_Node_Pool EvaluateLindenmayerOps(std::vector<Lindenmayer_Op> const& ops) {
    Tree_Node_Pool pool;
    std::stack<Execution_State> stack;
    Execution_State state = {};
    uint32_t uiRoot;
    float const flStep = 64.0f;

    state.flPitch = M_PI / 2.0f;
    pool.Allocate(uiRoot);
    state.uiNodeCurrent = uiRoot;

    for (uint32_t pc = 0; pc < ops.size(); pc++) {
        auto dir = GetDirectionVector(state);
        switch (ops[pc]) {
        case Lindenmayer_Op::Forward:
        {
            uint32_t nextNodeIdx;
            auto& nextNode = pool.Allocate(nextNodeIdx);
            auto& curNode = pool.GetNode(state.uiNodeCurrent);
            nextNode.vPosition = curNode.vPosition + flStep * dir;
            printf("Node: (%f, %f, %f)\n", nextNode.vPosition[0], nextNode.vPosition[1], nextNode.vPosition[2]);
            curNode.AddChild(nextNodeIdx);
            state.uiNodeCurrent = nextNodeIdx;
            break;
        }
        case Lindenmayer_Op::Push:
        {
            stack.push(state);
            break;
        }
        case Lindenmayer_Op::Pop:
        {
            state = stack.top();
            stack.pop();
            break;
        }
        case Lindenmayer_Op::Yaw_Pos:
        {
            state.flPitch += 0.785398163f;
            break;
        }
        case Lindenmayer_Op::Yaw_Neg:
        {
            state.flPitch -= 0.785398163f;
            break;
        }
        default:
            break;
        }
    }

    return pool;
}

#include <future>
#include <optional>
#include <variant>
#include <memory>

struct Future_Union_Mesh {
    using FM = std::future<Mesh_Builder::Optimized_Mesh>;
    using OT = std::unique_ptr<Future_Union_Mesh>;
    using OFM = std::optional<FM>;
    using V = std::variant <std::monostate, FM, OT>;

    V lhs;
    OT rhs;


    Mesh_Builder::Optimized_Mesh operator()(FM& x) {
        return x.get();
    }

    Mesh_Builder::Optimized_Mesh operator()(OT& x) {
        if (x != NULL) {
            return *x;
        } else {
            return {};
        }
    }

    Mesh_Builder::Optimized_Mesh operator()(std::monostate const&) {
        return {};
    }

    operator Mesh_Builder::Optimized_Mesh() {
        auto const x = std::visit(*this, lhs);
        if (rhs != NULL) {
            return x + *rhs;
        } else {
            return x;
        }
    }
};

void Union(Future_Union_Mesh& lhs, Future_Union_Mesh&& rhs) {
    auto x = std::make_unique<Future_Union_Mesh>(std::move(lhs));
    auto y = std::make_unique<Future_Union_Mesh>(std::move(rhs));
    lhs = Future_Union_Mesh { std::move(x), std::move(y) };
}

void Union(Future_Union_Mesh& lhs, Future_Union_Mesh::FM&& fm) {
    auto y = std::make_unique<Future_Union_Mesh>(std::move(lhs));
    lhs = Future_Union_Mesh{ std::move(fm), std::move(y) };
}

static Mesh_Builder::Optimized_Mesh ProcessNodes(Tree_Node_Pool const& tree, uint32_t const uiStart, uint32_t const uiBranch, uint32_t const uiEnd) {
    std::vector<lm::Vector4> points;

    // start node might have multiple children.
    // the code below won't handle that properly, so we add
    // that node here
    auto const& pStart = tree.GetNode(uiStart);
    auto const& pSecond = tree.GetNode(uiBranch);
    points.push_back(pStart.vPosition - (pSecond.vPosition - pStart.vPosition));
    points.push_back(pStart.vPosition);
    uint32_t uiCursor = uiBranch;
    uint32_t uiPrev = uiStart;
    while(1) {
        auto const& cur = tree.GetNode(uiCursor);
        points.push_back(cur.vPosition);

        if (uiCursor == uiEnd) {
            break;
        }

        assert(cur.unChildCount == 1);
        uiPrev = uiCursor;
        uiCursor = cur.aiChildren[0];
    };
    auto const& pEnd = tree.GetNode(uiEnd);
    auto const& pPenultimate = tree.GetNode(uiPrev);
    points.push_back(pEnd.vPosition + (pEnd.vPosition - pPenultimate.vPosition));

    printf("(%f, %f, %f) -> (%f, %f, %f)\n",
        pStart.vPosition[0], pStart.vPosition[1], pStart.vPosition[2],
        pEnd.vPosition[0], pEnd.vPosition[1], pEnd.vPosition[2]
        );

    Catmull_Rom_Composite<lm::Vector4> cr(points.size(), points.data());
    return MeshFromSpline(cr, [](auto i, auto const& p) { return 4.0f; });
}

static Future_Union_Mesh ProcessMultiNode(Tree_Node_Pool const& tree, uint32_t const uiNode) {
    Future_Union_Mesh ret;

    auto const& node = tree.GetNode(uiNode);
    // assert(node.unChildCount > 1);
    for (uint32_t uiChildOff = 0; uiChildOff < node.unChildCount; uiChildOff++) {
        auto const uiBranchHead = node.aiChildren[uiChildOff];
        auto uiCurrent = uiBranchHead;
        auto const* pCurrent = &tree.GetNode(uiCurrent);
        while (pCurrent->unChildCount == 1) {
            uiCurrent = pCurrent->aiChildren[0];
            pCurrent = &tree.GetNode(uiCurrent);
        }

        Union(ret, std::async(&ProcessNodes, tree, uiNode, uiBranchHead, uiCurrent));
        if (pCurrent->unChildCount > 1) {
            Union(ret, ProcessMultiNode(tree, uiCurrent));
        }
    }

    return ret;
}

static Mesh_Builder::Optimized_Mesh ProcessTree(Tree_Node_Pool const& tree) {
    auto const& root = tree.GetNode(0);
    assert(root.unChildCount > 0);

    return ProcessMultiNode(tree, 0);
}

int main(int argc, char** argv) {
    SDL_Init(SDL_INIT_EVERYTHING);
    GL_Renderer r;

    Lindenmayer_System::Alphabet const alphabet = {
        {'[', {Lindenmayer_Op::Push, Lindenmayer_Op::Yaw_Neg}},
        {']', {Lindenmayer_Op::Pop, Lindenmayer_Op::Yaw_Pos}},
        {'0', {Lindenmayer_Op::Forward}},
        {'1', {Lindenmayer_Op::Forward}},
    };

    Lindenmayer_System::Rule_Set const rules = {
        {'1', "11"},
        {'0', "1[0]0"},
    };

    Lindenmayer_System sys("0", alphabet, rules);
    auto const tree = EvaluateLindenmayerOps(sys.Iterate(3));

    if (r) {
        SDL_GL_SetSwapInterval(-1);
        gladLoadGLLoader(SDL_GL_GetProcAddress);

        if (glDebugMessageCallback) {
            glDebugMessageCallback(GLMessageCallback, 0);
        } else {
            printf("[ gfx ] BACKEND WARNING: no messages will be received from the driver!\n");
        }

        srand(time(NULL));

        /*
        lm::Vector4 const vertices[] = {
            {0.5f,  0.5f, 0.0f,},
            {0.5f, -0.5f, 0.0f,},
            {-0.5f,  0.5f, 0.0f,},
            {0.5f, -0.5f, 0.0f,},
            {-0.5f, -0.5f, 0.0f,},
            {-0.5f,  0.5f, 0.0f},
        };

        lm::Vector4 controlPoints[12];
        lm::Vector4 dir(0, 1, 0);
        controlPoints[0] = lm::Vector4();
        for (int i = 1; i < 12; i++) {
            float const dx = randf() * 2 - 1;
            float const dy = randf();
            float const dz = randf() * 2 - 1;
            dir = dir + lm::Vector4(dx, dy, dz);
            controlPoints[i] = controlPoints[i - 1] + 16 * dir;
        }

        Mesh_Builder mb;
        mb.PushTriangle(vertices[0], vertices[1], vertices[2]);
        mb.PushTriangle(vertices[3], vertices[4], vertices[5]);
        auto optmesh = mb.Optimize();

        Catmull_Rom_Composite<lm::Vector4> cr(12, controlPoints);
        auto optmesh2 = MeshFromSpline(cr, [=](size_t i, lm::Vector4 const& p) {
            return 16.0f;
        });
        auto asd = BuildModel(optmesh2);
        */

        auto asd = BuildModel(ProcessTree(tree));

        auto vsh = FromFileLoadShader<GL_VERTEX_SHADER>("generic.vsh.glsl");
        auto fsh = FromFileLoadShader<GL_FRAGMENT_SHADER>("generic.fsh.glsl");

        if (vsh && fsh) {
            auto builder = gl::Shader_Program_Builder();
            auto program = builder.Attach(vsh.value()).Attach(fsh.value()).Link();
            if (program) {
                auto hProgram = std::move(program.value());
                // Draw_Load_Unit dlu = { std::move(hProgram), BuildModel(optmesh) };
                Draw_Load_Unit dlu = { std::move(hProgram), std::move(asd) };
                RenderLoop(r, dlu);
            } else {
                printf("Failed to link shader program: %s\n", builder.Error());
            }
        } else {
            printf("Failed to load the generic shaders!\n");
        }
    }
    SDL_Quit();
    return 0;
}