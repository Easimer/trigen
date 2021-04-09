// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: arcball camera declaration
//

#pragma once

#include <memory>
#include <glm/mat4x4.hpp>

class Arcball_Camera {
public:
    virtual ~Arcball_Camera() {}
    virtual glm::mat4 get_view_matrix() = 0;

    virtual void set_screen_size(int x, int y) = 0;

    virtual void mouse_down(int x, int y) = 0;
    virtual void mouse_up(int x, int y) = 0;
    virtual bool mouse_move(int x, int y) = 0;
    virtual void mouse_wheel(int y) = 0;

    virtual void reset() = 0;

    virtual void get_look_at(glm::vec3 &eye, glm::vec3 &center) = 0;
};

std::unique_ptr<Arcball_Camera> create_arcball_camera();
