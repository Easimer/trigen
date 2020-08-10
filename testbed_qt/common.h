// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: common definitions
//

#pragma once

#if __cplusplus
#include <memory>
#include <functional>
#include <optional>

template<typename T>
using Unique_Ptr = std::unique_ptr<T>;

template<typename T>
using Fun = std::function<T>;

template<typename T>
using Optional = std::optional<T>;

#define GLM_ENABLE_EXPERIMENTAL

#include <cassert>
#include <cstdio>
#include <vector>
#include <unordered_map>
#include <glm/gtc/constants.hpp>
#include <glm/mat3x3.hpp>
#include <glm/vec3.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/matrix_decompose.hpp>

#include <KHR/khrplatform.h>
#include <glad/glad.h>

using Vec2 = glm::vec2;
using Vec3 = glm::vec3;
using Mat3 = glm::mat3;
using Mat4 = glm::mat4;
using Quat = glm::quat;
#else

#include <assert.h>
#include <stdio.h>
#include <KHR/khrplatform.h>
#include <glad/glad.h>

#endif /* __cplusplus */
