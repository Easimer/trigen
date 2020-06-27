// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: 
//

#pragma once

#include <cstdint>
#include <vector>
#include <trigen/glres.h>
#include <trigen/linear_math.h>

namespace rq {
    enum Render_Command_Kind : uint32_t {
        k_unRCInvalid = 0,
        k_unRCRenderSubQueue,
        k_unRCChangeProgram,
        k_uNRCDrawElementModel,
        k_unRCDebugNote,
        k_unRCMax
    };

    struct Render_Command;
    class Render_Queue;

    struct Render_Sub_Queue_Params {
        Render_Queue* queue;
    };

    struct Draw_Element_Model_Params {
        // gl::Weak_Resource_Reference<gl::VAO> vao;
        GLuint vao;
        size_t elements;
        // lm::Vector4 position;
        float position[3];
    };

    struct Change_Program_Params {
        // gl::Weak_Resource_Reference<gl::Shader_Program> program;
        GLuint program;
    };

    struct Debug_Note_Params {
        char msg[64];
    };

    struct Render_Command {
        Render_Command_Kind kind;
        union {
            Draw_Element_Model_Params draw_element_model;
            Change_Program_Params change_program;
            Render_Sub_Queue_Params render_sub_queue;
            Debug_Note_Params note;
        };
    };

    class Render_Queue {
    public:
        void Add(Render_Command const& cmd) {
            buffer.push_back(cmd);
        }

        void Add(Render_Command&& cmd) {
            buffer.push_back(std::move(cmd));
        }

        void Clear() {
            buffer.clear();
        }

        // cpp foreach
        auto begin() { return buffer.begin(); }
        auto end() { return buffer.end(); }
    private:
        std::vector<Render_Command> buffer;
    };

    // Make a Draw_Element_Model_Params in a type-safe manner
    inline Draw_Element_Model_Params MakeDrawElementModelParams(
        gl::Weak_Resource_Reference<gl::VAO> vao,
        size_t elements,
        lm::Vector4 const& pos
    ) {
        Draw_Element_Model_Params ret = { vao, elements };
        for (int i = 0; i < 3; i++) ret.position[i] = pos[i];
        return ret;
    }

    // Make a Change_Program_Params in a type-safe manner
    inline Change_Program_Params MakeChangeProgramParams(
        gl::Weak_Resource_Reference<gl::Shader_Program> program
    ) {
        return { program };
    }
}
