// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: arcball camera declaration
//

#pragma once

#include "common.h"

class Arcball_Camera {
public:
    virtual ~Arcball_Camera() {}
    virtual Mat4 get_view_matrix() = 0;
    virtual void set_screen_size(int x, int y) = 0;

    virtual void mouse_down(int x, int y) = 0;
    virtual void mouse_up(int x, int y) = 0;
    virtual bool mouse_move(int x, int y) = 0;
    virtual void mouse_wheel(int y) = 0;

    virtual void reset() = 0;
};

Unique_Ptr<Arcball_Camera> create_arcball_camera();
