// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#pragma once

#include <r_renderer.h>
#include <r_queue.h>

class Render_Grid : public gfx::IRender_Command {
private:
    virtual void execute(gfx::IRenderer *renderer) override;
};

class Render_Model : public gfx::IRender_Command {
public:
    Render_Model(gfx::Model_ID model, gfx::Texture_ID diffuse, gfx::Transform const &transform);

    glm::vec4 &tint() { return _tintColor; }

    void execute(gfx::IRenderer *renderer) override;
private:
    gfx::Model_ID _model;
    gfx::Texture_ID _diffuse;
    gfx::Transform _transform;
    glm::vec4 _tintColor;
};

class Render_Untextured_Model : public gfx::IRender_Command {
public:
    Render_Untextured_Model(gfx::Model_ID model, gfx::Transform const &transform);

    glm::vec4 &tint() { return _tintColor; }

    void execute(gfx::IRenderer *renderer) override;
private:
    gfx::Model_ID _model;
    gfx::Transform _transform;
    glm::vec4 _tintColor;
};

class Load_Texture_Command : public gfx::IRender_Command {
public:
    Load_Texture_Command(std::optional<gfx::Texture_ID> *handle, void const *image, size_t image_len);

    void execute(gfx::IRenderer *renderer) override;

private:
    std::optional<gfx::Texture_ID> *_handle;
    void const *_image;
    size_t _image_len;
};

class Destroy_Model_Command : public gfx::IRender_Command {
public:
    Destroy_Model_Command(gfx::Model_ID id);

    void execute(gfx::IRenderer *renderer) override;
private:
    gfx::Model_ID _id;
};

class Fetch_Camera_Matrices : public gfx::IRender_Command {
public:
    Fetch_Camera_Matrices(glm::mat4 *out_viewMatrix, glm::mat4 *out_projectionMatrix);

protected:
    void execute(gfx::IRenderer *renderer) override;
    glm::mat4 *_viewMatrix;
    glm::mat4 *_projMatrix;
};
