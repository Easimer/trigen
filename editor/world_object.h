// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#pragma once

#include <utils/Entity.h>

class World_Object {
public:
	virtual ~World_Object() = default;

protected:
	utils::Entity _entity;
};