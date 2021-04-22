// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: Qt wrapper for the World class
//

#pragma once
#include "world.h"

class QWorld : public QObject {
    Q_OBJECT;
public:
	Entity_Handle createEntity() {
		auto handle = _world.createEntity();
		emit entityAdded(handle);
		return handle;
	}

	bool removeEntity(Entity_Handle handle) {
		auto ret = _world.removeEntity(handle);

		if (ret) {
			emit entityRemoved(handle);
		}

		return ret;
	}

	template<typename T, typename ...Args>
	T *attachComponent(Entity_Handle ent, Args... args) {
		auto ret = _world.attachComponent<T, Args...>(ent, args...);

        return ret;
	}

	template<typename T>
    std::unordered_map<Entity_Handle, T> &getMapForComponent() {
        return _world.getMapForComponent<T>();
    }

	template<typename T>
    std::unordered_map<Entity_Handle, T> const &getMapForComponent() const {
        return _world.getMapForComponent<T>();
    }

	int numEntities() const {
		return _world.numEntities();
	}

	bool exists(Entity_Handle ent) const {
		return _world.exists(ent);
	}

signals:
	void entityAdded(Entity_Handle handle);
	void entityRemoved(Entity_Handle handle);

private:
    World _world;
};
