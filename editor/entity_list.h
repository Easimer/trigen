// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#pragma once

#include <cassert>
#include <memory>
#include <QAbstractItemModel>

class World;

class Entity_List_Model : public QAbstractListModel {
    Q_OBJECT;
public:
    Entity_List_Model(QObject *parent = nullptr) : QAbstractListModel(parent), _world(nullptr) {
    }

    Q_INVOKABLE void setCurrentWorld(World const *world);

    Q_INVOKABLE int rowCount(const QModelIndex &parent = QModelIndex()) const override;
    Q_INVOKABLE QVariant data(const QModelIndex &index, int role = Qt::DisplayRole) const override;
    Q_INVOKABLE QVariant headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const override;

private:
    World const *_world;
};
