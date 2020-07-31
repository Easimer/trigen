// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: common declarations
//

#pragma once

#include <vector>
#include <mutex>
#include <queue>
#include <unordered_map>
#include <functional>

#include <glm/mat3x3.hpp>
#include <glm/vec3.hpp>
#include <glm/gtc/quaternion.hpp>

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
