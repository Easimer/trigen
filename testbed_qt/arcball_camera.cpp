// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: arcball camera
//

//
// Based on https://github.com/Twinklebear/arcball-cpp
// Copyright (c) 2016 Will Usher
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//

#include "common.h"
#include "arcball_camera.h"
#include <optional>

template<typename T> using Optional = std::optional<T>;

static Quat ScreenToArcball(Vec2 const& p);

#define ARCBALL_PARANOID(q) glm::normalize(q)

class Arcball_Camera_Impl : public Arcball_Camera {
public:
    Arcball_Camera_Impl() {
        screen_size = Vec2(0, 0);

        reset();
    }
private:
    void rotate(Vec2 prev, Vec2 cur) {
        cur = glm::clamp(cur, Vec2(-1, -1), Vec2(1, 1));
        prev = glm::clamp(prev, Vec2(-1, -1), Vec2(1, 1));

        auto cur_ball = ARCBALL_PARANOID(ScreenToArcball(cur));
        auto prev_ball = ARCBALL_PARANOID(ScreenToArcball(prev));

        rotation = ARCBALL_PARANOID(cur_ball * prev_ball * rotation);

        update_camera();
    }

    void reset() override {
        auto eye = Vec3(0, 0, 1);
        auto center = Vec3();
        auto up = Vec3(0, 1, 0);
        auto dir = center - eye;
        auto z_axis = glm::normalize(dir);
        auto x_axis = glm::normalize(glm::cross(z_axis, glm::normalize(up)));
        auto y_axis = glm::normalize(glm::cross(x_axis, z_axis));

        x_axis = glm::normalize(glm::cross(z_axis, y_axis));

        center_translation = -center;
        translation = Vec3(0, 0, -glm::length(dir));
        rotation = glm::normalize(glm::quat_cast(glm::transpose(Mat3(x_axis, y_axis, -z_axis))));

        update_camera();
    }

     void mouse_down(int x, int y) override {
        mouse_position = TransformCursorPosition(x, y);
     }

     void mouse_up(int x, int y) override {
         mouse_position.reset();
     }

     bool mouse_move(int x, int y) override {
         if (mouse_position) {
             Vec2 cur = TransformCursorPosition(x, y);
             rotate(*mouse_position, cur);
             mouse_position = cur;
             return true;
         } else {
             return false;
         }
     }

     void mouse_wheel(int y) override {
         auto wheel = y;
         float z = translation.z;
         if (wheel > 0) {
             z *= 2.0f;
         } else {
             z /= 2.0f;
         }

         if (z > -0.1) z = -0.1;
         translation.z = z;
         update_camera();
     }

    Vec2 TransformCursorPosition(int x0, int y0) {
        auto x1 = 2 * (x0 / screen_size.x) - 1;
        auto y1 = 2 * (1 - y0 / screen_size.y) - 1;

        return { x1, y1 };
    }

    Mat4 get_view_matrix() override {
        return camera;
    }

    void update_camera() {
        camera = glm::translate(translation) * Mat4(ARCBALL_PARANOID(rotation)) * glm::translate(center_translation);
    }

    void set_screen_size(int x, int y) override {
        screen_size = Vec2(x, y);
    }

    Vec2 screen_size;
    Vec3 center_translation, translation;
    Quat rotation;

    Mat4 camera;

    Optional<Vec2> mouse_position;
};

Unique_Ptr<Arcball_Camera> create_arcball_camera() {
    return std::make_unique<Arcball_Camera_Impl>();
}

static Quat ScreenToArcball(Vec2 const& p) {
    float dist = glm::dot(p, p);
    if (dist <= 1.f) {
        return Quat(0.0, p.x, p.y, glm::sqrt(1.f - dist));
    } else {
        const Vec2 proj = glm::normalize(p);
        return Quat(0.0, proj.x, proj.y, 0.f);
    }
}
