// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#include "stdafx.h"
#include "filament_factory.h"

#include <filament/Scene.h>

Filament_Factory::Filament_Factory() : _engine(nullptr), _commonSkybox(nullptr) {
}

Filament_Factory::Filament_Factory(filament::Engine *engine) :
    _engine(engine),
    _commonSkybox(nullptr) {
}

filament::Ptr<filament::Scene> Filament_Factory::createScene() {
    auto ret = filament::Ptr(_engine->createScene(), _engine);

    if (_commonSkybox != nullptr) {
        ret->setSkybox(_commonSkybox);
    }

    return ret;
}

void Filament_Factory::setCommonSkybox(filament::Skybox *skybox) {
    _commonSkybox = skybox;
}
