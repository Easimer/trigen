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

private:
    filament::Engine *_engine;
};
