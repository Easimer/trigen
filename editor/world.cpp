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
    optEnt.emplace(Entity{ });

    return ret;
}

bool World::removeEntity(Entity_Handle handle) {
    if (handle >= _entities.size() || !_entities[handle].has_value()) {
        return false;
    }

#define REMOVE_COMPONENT_FOR_ENTITY(typeName, dataMember) dataMember.erase(handle);
    FOREACH_COMPONENT(REMOVE_COMPONENT_FOR_ENTITY);

    _entities[handle].reset();

    return true;
}

bool World::exists(Entity_Handle ent) const {
    if (ent < _entities.size()) {
        return _entities[ent].has_value();
    }

    return false;
}
