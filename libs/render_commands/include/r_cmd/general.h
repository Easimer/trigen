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

