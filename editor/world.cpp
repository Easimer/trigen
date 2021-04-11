// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#include "stdafx.h"
#include "world.h"

Entity_Handle World::createEntity() {
    std::optional<Entity> *slot = nullptr;
    Entity_Handle ret;

    for (Entity_Handle i = 0; i < _entities.size(); i++) {
        if (!_entities[i].has_value()) {
            slot = &_entities[i];
            ret = i;
            break;
        }
    }

    if (slot == nullptr) {
        _entities.push_back({});
        slot = &_entities.back();
        ret = _entities.size() - 1;
    }

    auto &optEnt = *slot;
    optEnt.emplace(Entity{ utils::EntityManager::get().create() });

    _scene->addEntity(optEnt->entity);

    return ret;
}
