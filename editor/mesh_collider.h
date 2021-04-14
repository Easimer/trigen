// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#pragma once

#include <softbody.h>

struct Collider_Render_Buffers {
};

class IMesh_Collider {
public:
	virtual sb::Mesh_Collider *collider() = 0;
	virtual void fillRenderBuffers(Collider_Render_Buffers *buffers) = 0;
};
