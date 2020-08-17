// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: meshgen window implementation
//

#include "common.h"
#include "wnd_meshgen.h"
#include "softbody_renderer.h"

using namespace std::placeholders;

Window_Meshgen::Window_Meshgen(
    sb::Unique_Ptr<sb::ISoftbody_Simulation>& simulation,
    QMainWindow* parent) :
    QDialog(parent),
    simulation(simulation) {
}

class Render_Test : public gfx::IRender_Command {
    void execute(gfx::IRenderer* renderer) override {
        Vec3 line[2] = { { -4, 2, 0 }, { 4, 2, 0 } };
        renderer->draw_lines(line, 1, Vec3(0, 0, 0), Vec3(1, 1, 1), Vec3(1, 1, 1));
    }
};

void Window_Meshgen::render(gfx::Render_Queue* rq) {
    auto c = rq->allocate<Render_Test>();
    new(c) Render_Test();
    rq->push(c);
}
