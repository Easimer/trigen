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
    /**
     * Render command interface.
     *
     * Subclass this to create new kinds of render commands.
     * Use gfx::allocate_command_and_initialize<T>(rq, ...) to put a new
     * command into a render queue.
     * The renderer will call the `execute` method of your command, passing
     * in a pointer to the renderer itself.
     */
    class IRender_Command {
    public:
        virtual ~IRender_Command() = default;
        virtual void execute(IRenderer* renderer) = 0;
    };

    /**
     * An arena allocator intended to be used only by Render_Queue.
     */
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

    /**
     * Render queue. Stores render commands to be executed.
     */
    class Render_Queue {
    public:
        /**
         * Initialize the queue to a fixed size.
         * @param queueSize size in bytes
         *
         * @note exceeding the size limit will cause an assertion failure on
         * debug builds and a crash (if you're lucky) on release builds!
         */
        Render_Queue(size_t queueSize) : m_arena(queueSize) {
        }

        /**
         * Allocates a new command BUT DOES NOT ENQUEUE IT!
         * @param T type of the render command; should be a subclass of
         * IRender_Command.
         * @return a pointer to the allocated command
         *
         * @note Don't call this method directly if you don't have to. Use
         * gfx::allocate_command_and_initialize instead!
         */
        template<typename T>
        T* allocate() {
            return m_arena.allocate<T>();
        }

        /**
         * Enqueues a command allocated using `allocate`.
         * @param pointer to a command returned by `allocate`.
         *
         * @note If `cmd` wasn't allocated in this render queue you WILL
         * experience crashes!
         *
         * @note Don't call this method directly if you don't have to. Use
         * gfx::allocate_command_and_initialize instead!
         */
        void push(IRender_Command* cmd) {
            m_commands.push_back(cmd);
        }

        /**
         * Clears the render queue.
         */
        void clear() {
            for (auto &cmd : m_commands) {
                cmd->~IRender_Command();
            }

            m_commands.clear();
            m_arena.clear();
        }

        void execute(IRenderer* renderer, bool do_clear = true) {
            for (auto& cmd : m_commands) {
                cmd->execute(renderer);
            }

            if (do_clear) {
                clear();
            }
        }
    private:
        std::vector<IRender_Command*> m_commands;
        Render_Command_Arena m_arena;
    };

    /**
     * Allocates a new command in a render queue and initializes it.
     *
     * @param T type of the command
     * @param rq Destination queue
     * @param args Constructor arguments to pass to the command
     * @return Pointer to the newly created command
     */
    template<typename T, class ... Arg>
    T* allocate_command_and_initialize(gfx::Render_Queue* rq, Arg ... args) {
        auto cmd = rq->allocate<T>();
        new(cmd) T(args...);
        rq->push(cmd);
        return cmd;
    }
}
