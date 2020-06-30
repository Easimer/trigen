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
#include <execution>
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

#include <KHR/khrplatform.h>
#include <glad/glad.h>

using Vec3 = glm::vec3;
using Mat3 = glm::mat3;
using Quat = glm::quat;

template<typename T>
using Vector = std::vector<T>;

template<typename K, typename V>
using Map = std::unordered_map<K, V>;

template<typename T>
using Dequeue = std::deque<T>;

template<typename T>
using Fun = std::function<T>;

using Mutex = std::mutex;
using Lock_Guard = std::lock_guard<std::mutex>;
#else

#include <assert.h>
#include <stdio.h>
#include <KHR/khrplatform.h>
#include <glad/glad.h>

#endif /* __cplusplus */
