// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: common declarations
//

#pragma once

#include <softbody.h>
#include "types.h"
#include "collision_constraint.h"

struct Mesh_Collider_Slot {
    bool used;

    Mat4 transform;
    size_t triangle_count;
    Vector<uint64_t> vertex_indices;
    Vector<uint64_t> normal_indices;
    Vector<Vec3> vertices;
    Vector<Vec3> normals;
};

struct SDF_Slot {
    bool used;
    sb::sdf::ast::Expression<float>* expr;
    sb::sdf::ast::Sample_Point* sp;
};

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
