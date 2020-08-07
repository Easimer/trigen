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
#include <unordered_set>

#include <glm/mat3x3.hpp>
#include <glm/vec3.hpp>
#include <glm/gtc/quaternion.hpp>

using Vec3 = glm::vec3;
using Vec4 = glm::vec4;
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

template<typename T>
using Set = std::unordered_set<T>;

struct Collision_Constraint {
    unsigned pidx;
    Vec3 intersect, normal;
};

using index_t = typename std::make_signed<size_t>::type;

struct System_State {
    Vector<Vec3> bind_pose;
    // Position in the previous frame
    Vector<Vec3> position;
    // Position in the current frame
    Vector<Vec3> predicted_position;

    // Particle velocities
    Vector<Vec3> velocity;
    // Particle angular velocities
    Vector<Vec3> angular_velocity;

    // Particle sizes
    Vector<Vec3> size;

    // Particle orientations in the last frame
    Vector<Quat> orientation;
    // Particle orientations in the current frame
    Vector<Quat> predicted_orientation;

    // Particle densities
    Vector<float> density;
    // Particle ages
    //Vector<float> age;
    Map<unsigned, Vector<unsigned>> edges;

    Vector<Vec3> bind_pose_center_of_mass;
    Vector<Mat3> bind_pose_inverse_bind_pose;

    struct SDF_Slot {
        bool used;
        std::function<float(glm::vec3 const&)> fun;
    };

    Vector<SDF_Slot> colliders_sdf;
    Vector<Collision_Constraint> collision_constraints;

    // For debug visualization only
    Vector<Vec3> center_of_mass;
    Vector<Vec3> goal_position;

    Set<unsigned> fixed_particles;
};
