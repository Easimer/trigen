// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: debug visualizer 
//

#pragma once

#include "softbody.h"

class ITestbed_Debug_Visualizer : public sb::IDebug_Visualizer {
public:
    virtual void execute(gfx::IRenderer *renderer) = 0;
};

Unique_Ptr<ITestbed_Debug_Visualizer> make_debug_visualizer();

class Subqueue_Render_Command : public gfx::IRender_Command {
public:
    Subqueue_Render_Command(ITestbed_Debug_Visualizer *visualizer) : _visualizer(visualizer) {
    }

    void execute(gfx::IRenderer *renderer) override {
        _visualizer->execute(renderer);
    }

private:
    ITestbed_Debug_Visualizer *_visualizer;
};

// HOW TO USE THE DEBUG VISUALIZER
// A sb::ISoftbody_Simulation instance can take a debug visualizer instance
// through the set_debug_visualizer method.
// Call make_debug_visualizer() to create a new visualizer, use the method
// mentioned above.
// Now you should have a GLViewport instance somewhere. It will call each frame
// that lambda that you've set as the render queue filler. That lambda should
// put a Subqueue_Render_Command into the queue. This will execute the subqueue
// stored in the visualizer.
