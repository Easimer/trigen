// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: softbody simulation library
//

#pragma once

#if __cplusplus
#define GLM_ENABLE_EXPERIMENTAL

#include <cassert>
#include <cstdio>
#include <vector>
#include <unordered_map>
#include <thread>
#include <algorithm>
#include <optional>
#include <queue>
#include <functional>
#include <glm/gtc/constants.hpp>
#include <glm/mat3x3.hpp>
#include <glm/vec3.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/matrix_decompose.hpp>

#include "common.h"

#else

#include <assert.h>
#include <stdio.h>

#endif /* __cplusplus */
