// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#include "stdafx.h"
#include "entity_list.h"
#include "world.h"

Q_INVOKABLE int Entity_List_Model::rowCount(const QModelIndex &parent) const {
    if (_world == nullptr) {
        return 0;
    }

    return _world->numEntities();
}

Q_INVOKABLE QVariant Entity_List_Model::data(const QModelIndex &index, int role) const {
    if (_world == nullptr) {
        return QVariant();
    }

    Entity_Handle entHandle = index.row();
    if (!_world->exists(entHandle)) {
        return QVariant();
    }

    if (role == Qt::DisplayRole) {
        return QString("Entity #") + QString::number(index.row());
    } else {
        return QVariant();
    }
}

Q_INVOKABLE QVariant Entity_List_Model::headerData(int section, Qt::Orientation orientation, int role) const {
    return "section";
}

Q_INVOKABLE void Entity_List_Model::setCurrentWorld(World const *world) {
    _world = world;
    emit dataChanged(createIndex(0, 0), createIndex(_world->numEntities() - 1, 0), { Qt::DisplayRole });
}
