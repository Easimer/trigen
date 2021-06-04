// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: common types
//

#pragma once

#include <deque>
#include <functional>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <glm/gtc/quaternion.hpp>
#include <glm/mat3x3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

using Vec3 = glm::vec3;
using Vec4 = glm::vec4;
using Mat3 = glm::mat3;
using Mat4 = glm::mat4;
using Quat = glm::quat;

template <typename T>
using Vector = std::vector<T>;

template <typename K, typename V>
using Map = std::unordered_map<K, V>;

template <typename T>
using Dequeue = std::deque<T>;

template <typename T>
using Fun = std::function<T>;

using Mutex = std::mutex;
using Lock_Guard = std::lock_guard<std::mutex>;

template <typename T>
using Set = std::unordered_set<T>;

using index_t = typename std::make_signed<size_t>::type;