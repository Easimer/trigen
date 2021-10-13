// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: 
//

#pragma once

#include "async_image_loader.h"

class IApplication {
public:
    virtual ~IApplication() { }

    virtual IAsync_Image_Loader *
    ImageLoader()
        = 0;

    virtual topo::IInstance *
    Renderer()
        = 0;

    virtual Trigen_Session
    Simulation()
        = 0;

    virtual void
    SetSimulation(Trigen_Session sim)
        = 0;

    virtual uv_loop_t *
    Loop()
        = 0;

    virtual void
    OnLeafTextureLoaded(topo::Texture_ID texture)
        = 0;
    virtual void
    OnInputTextureLoaded()
        = 0;

    virtual void
    OnTreeVisualsReady()
        = 0;

    virtual void
    OnSimulationStepOver()
        = 0;
};
