// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: objscan API
//

#pragma once

struct objscan_connection {
	long long idx0, idx1;
};

struct objscan_position {
	float x, y, z, w;
};

struct objscan_result {
	long long particle_count;
	objscan_position* positions;
	long long connection_count;
	objscan_connection* connections;
};

bool objscan_from_obj_file(objscan_result* res, char const* path);
void objscan_free_result(objscan_result* res);