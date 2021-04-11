// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#pragma once

#include "filament_wrapper.h"

class Filament_Factory {
public:
    Filament_Factory();
    Filament_Factory(filament::Engine *engine);
    filament::Ptr<filament::Scene> createScene();
    void setCommonSkybox(filament::Skybox *skybox);

private:
    filament::Engine *_engine;
    filament::Skybox *_commonSkybox;
};
