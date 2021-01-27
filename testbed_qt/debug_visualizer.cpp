// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: debug visualizer 
//

#include "common.h"
#include "debug_visualizer.h"

static constexpr size_t debug_visualizer_queue_size = 8 * 4096;

class Command_Draw_Line : public gfx::IRender_Command {
public:
    Command_Draw_Line(glm::vec3 const &start, glm::vec3 const &end, glm::vec3 const &colStart, glm::vec3 const &colEnd)
        : _points{ start, end }, _colors{ colStart, colEnd } {
    }
private:
    virtual void execute(gfx::IRenderer *renderer) override {
        renderer->draw_lines(_points + 0, 1, Vec3(0, 0, 0), _colors[0], _colors[1]);
    }

    glm::vec3 _points[2];
    glm::vec3 _colors[2];
};

class Debug_Visualizer : public ITestbed_Debug_Visualizer {
public:
    Debug_Visualizer() : _rq(debug_visualizer_queue_size) {
    }

    void new_frame() override {
        _rq.clear();
    }

    void draw_line(glm::vec3 start, glm::vec3 end) override {
        draw_line(start, end, Vec3(.5f, .5f, .5f), Vec3(1, 1, 1));
    }

    void draw_intersection(
        glm::vec3 const &start, glm::vec3 const &thru,
        glm::vec3 const &xp,
        glm::vec3 const &v0, glm::vec3 const &v1, glm::vec3 const &v2,
        glm::vec3 const &normal) override {
        // Ray 
        draw_line(start, thru, Vec3(0.5f, 0.5f, 1), Vec3(1, 1, 1));
        // XP
        draw_line(xp + Vec3(-1, 0, 0), xp + Vec3(1, 0, 0), Vec3(1, 0.125f, 0.125f), Vec3(1, 0.125f, 0.125f));
        draw_line(xp + Vec3(0, 0, -1), xp + Vec3(0, 0, 1), Vec3(1, 0.125f, 0.125f), Vec3(1, 0.125f, 0.125f));
        // Triangle
        draw_line(v0, v1, Vec3(0, 1, 0), Vec3(0, 1, 0));
        draw_line(v0, v2, Vec3(0, 1, 0), Vec3(0, 1, 0));
        draw_line(v1, v2, Vec3(0, 1, 0), Vec3(0, 1, 0));
        // Triangle normal
        auto center = (v0 + v1 + v2) / 3.0f;
        draw_line(center, center + normal, Vec3(0.1, 0.8, 0.1), Vec3(0.1, 0.8, 0.1));
    }

    void draw_line(glm::vec3 const &start, glm::vec3 const &end, glm::vec3 const &colStart, glm::vec3 const &colEnd) {
        gfx::allocate_command_and_initialize<Command_Draw_Line>(&_rq, start, end, colStart, colEnd);
    }

    void execute(gfx::IRenderer *renderer) override {
        _rq.execute(renderer, /* do_clear: */ false);
    }

private:
    gfx::Render_Queue _rq;
};

Unique_Ptr<ITestbed_Debug_Visualizer> make_debug_visualizer() {
    return std::make_unique<Debug_Visualizer>();
}
