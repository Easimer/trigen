// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: 
//

#pragma once

#include <cstdint>
#include <vector>
#include <memory>

#include "r_renderer.h"

namespace gfx {
    class IRender_Command {
    public:
        virtual void release() {}
        virtual void execute(IRenderer* renderer) = 0;
    };

    class Render_Command_Arena {
    public:
        Render_Command_Arena(size_t size) :
            m_buffer(std::make_unique<uint8_t[]>(size)),
            m_size(size),
            m_ptr(0)
        {
        }

        template<typename T>
        T* allocate() {
            auto const siz = sizeof(T);
            assert(m_ptr + siz < m_size);
            auto ret = (T*)(m_buffer.get() + m_ptr);
            m_ptr += siz;
            return ret;
        }

        void clear() {
            m_ptr = 0;
        }

    private:
        std::unique_ptr<uint8_t[]> m_buffer;
        size_t m_size;
        size_t m_ptr;
    };

    class Render_Queue {
    public:
        Render_Queue(size_t queueSize) : m_arena(queueSize) {
        }

        template<typename T>
        T* allocate() {
            return m_arena.allocate<T>();
        }

        void push(IRender_Command* cmd) {
            m_commands.push_back(cmd);
        }

        void execute(IRenderer* renderer) {
            for (auto& cmd : m_commands) {
                cmd->execute(renderer);
            }
            for (auto& cmd : m_commands) {
                cmd->release();
            }
            m_commands.clear();
            m_arena.clear();
        }
    private:
        std::vector<IRender_Command*> m_commands;
        Render_Command_Arena m_arena;
    };
}
