// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#pragma once

#include <cassert>
#include <memory>
#include <QAbstractItemModel>
#include "world_qt.h"

class Entity_List_Model : public QAbstractListModel {
    Q_OBJECT;
public:
    Entity_List_Model(QObject *parent = nullptr) : QAbstractListModel(parent), _world(nullptr) {
    }

    Q_INVOKABLE void setCurrentWorld(QWorld const *world);

    Q_INVOKABLE int rowCount(const QModelIndex &parent = QModelIndex()) const override;
    Q_INVOKABLE QVariant data(const QModelIndex &index, int role = Qt::DisplayRole) const override;
    Q_INVOKABLE QVariant headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const override;

protected slots:
    void entityAdded(Entity_Handle handle);
    void entityRemoved(Entity_Handle handle);

private:
    QWorld const *_world;
};
