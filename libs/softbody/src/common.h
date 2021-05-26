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
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <glm/gtc/quaternion.hpp>

#include <softbody.h>

using Vec3 = glm::vec3;
using Vec4 = glm::vec4;
using Mat3 = glm::mat3;
using Mat4 = glm::mat4;
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
    Vec4 intersect, normal;
    float depth;
};

using index_t = typename std::make_signed<size_t>::type;

struct System_State {
    Vector<Vec4> bind_pose;
    // Position in the previous frame
    Vector<Vec4> position;
    // Position in the current frame
    Vector<Vec4> predicted_position;

    // Particle velocities
    Vector<Vec4> velocity;
    // Particle angular velocities
    Vector<Vec4> angular_velocity;

    // Particle sizes
    Vector<Vec4> size;

    // Particle orientations in the last frame
    Vector<Quat> orientation;
    // Particle orientations in the current frame
    Vector<Quat> predicted_orientation;

    // Particle densities
    Vector<float> density;
    // Particle ages
    //Vector<float> age;
    Map<index_t, Vector<index_t>> edges;

    Vector<Vec4> bind_pose_center_of_mass;
    Vector<Mat4> bind_pose_inverse_bind_pose;

    struct SDF_Slot {
        bool used;
        sb::sdf::ast::Expression<float>* expr;
        sb::sdf::ast::Sample_Point* sp;
    };

    struct Mesh_Collider_Slot {
        bool used;

        Mat4 transform;
        size_t triangle_count;
        Vector<uint64_t> vertex_indices;
        Vector<uint64_t> normal_indices;
        Vector<Vec3> vertices;
        Vector<Vec3> normals;
    };

    Vector<SDF_Slot> colliders_sdf;
    Vector<Mesh_Collider_Slot> colliders_mesh;
    Vector<Collision_Constraint> collision_constraints;

    // For debug visualization only
    Vector<Vec4> center_of_mass;
    Vector<Vec4> goal_position;

    Set<index_t> fixed_particles;

    Vec4 global_center_of_mass;
    Vector<Vec4> internal_forces;

    Vec4 light_source_direction;
};

class ILogger {
public:
    virtual ~ILogger() = default;


    virtual void log(sb::Debug_Message_Source s, sb::Debug_Message_Type t, sb::Debug_Message_Severity l, char const* fmt, ...) = 0;
};

enum class Collider_Handle_Kind {
    SDF = 0,
    Mesh,
    Max
};


static sb::ISoftbody_Simulation::Collider_Handle make_collider_handle(Collider_Handle_Kind kind, size_t index) {
    static_assert(sizeof(ret) == 8, "Size of a collider handle was not 8 bytes.");
    assert((index & 0xF000'0000'0000'0000) == 0);
    assert((int)kind < 256);
    auto kindNibble = (size_t)kind;
    // Encode the type of the handle in the most significant nibble
    return (index & 0x0FFF'FFFF'FFFF'FFFF) | (kindNibble << 60);
}

static void decode_collider_handle(sb::ISoftbody_Simulation::Collider_Handle handle, Collider_Handle_Kind &kind, size_t &index) {
    auto kindNibble = (handle >> 60) & 0x0F;
    assert(kindNibble < (int)Collider_Handle_Kind::Max);
    kind = (Collider_Handle_Kind)kindNibble;
    index = (handle & 0x0FFF'FFFF'FFFF'FFFF);
}
