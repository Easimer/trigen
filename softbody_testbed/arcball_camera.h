// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: arcball camera declaration
//

#pragma once

#include "events.h"

class Arcball_Camera : public IEvent_Handler {
public:
    virtual void release() = 0;
    virtual Mat4 get_view_matrix() = 0;
    virtual void set_screen_size(int x, int y) = 0;
};

Arcball_Camera* create_arcball_camera();
