// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: UV inspector
//

#pragma once

#include <memory>

#include <glm/vec2.hpp>

#if UV_INSPECTOR_BUILDING
#define UV_INSPECTOR_EXPORT TRIGEN_DLLEXPORT
#else
#define UV_INSPECTOR_EXPORT TRIGEN_DLLIMPORT
#endif

namespace uv_inspector {
UV_INSPECTOR_EXPORT int
inspect(glm::vec2 const *texCoords, unsigned const *indices, int count);
}
