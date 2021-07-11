// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: 
//

#include "render_queue.h"

namespace topo {

void
Render_Queue::Submit(Renderable_ID renderable, Transform const &transform) {
    _commands.push_back({ renderable, transform });
}
void
Render_Queue::AddLight(glm::vec4 const &color, Transform const &transform) {
    _lights.push_back({ color, transform });
}

}
