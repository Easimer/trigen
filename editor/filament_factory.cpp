// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#include "stdafx.h"
#include "filament_factory.h"

Filament_Factory::Filament_Factory() : _engine(nullptr) {
}

Filament_Factory::Filament_Factory(filament::Engine *engine) : _engine(engine) {
}

filament::Ptr<filament::Scene> Filament_Factory::createScene() {
    return filament::Ptr(_engine->createScene(), _engine);
}
