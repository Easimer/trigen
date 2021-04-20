// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#include "stdafx.h"
#include "entity_list.h"
#include "world_qt.h"

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

Q_INVOKABLE void Entity_List_Model::setCurrentWorld(QWorld const *world) {
    if (_world != nullptr) {
        disconnect(_world, &QWorld::entityAdded, this, &Entity_List_Model::entityAdded);
        disconnect(_world, &QWorld::entityRemoved, this, &Entity_List_Model::entityRemoved);
    }
    _world = world;

    connect(_world, &QWorld::entityAdded, this, &Entity_List_Model::entityAdded);
    connect(_world, &QWorld::entityRemoved, this, &Entity_List_Model::entityRemoved);

    emit dataChanged(createIndex(0, 0), createIndex(_world->numEntities() - 1, 0), { Qt::DisplayRole });
}

void Entity_List_Model::entityAdded(Entity_Handle handle) {
    emit dataChanged(createIndex(handle, 0), createIndex(handle, 0), { Qt::DisplayRole });
}

void Entity_List_Model::entityRemoved(Entity_Handle handle) {
    emit dataChanged(createIndex(handle, 0), createIndex(handle, 0), { Qt::DisplayRole });
}
