// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: 
//

System_State : {
    @doc("Initial position")
    bind_pose: Vec4,

    @doc("Position in the previous frame")
    position: Vec4,

    @doc("Position in the current frame")
    predicted_position: Vec4,

    @doc("Velocity in the previous frame")
    velocity: Vec4,

    @doc("Angular velocity in the previous frame")
    angular_velocity: Vec4,

    @doc("Particle extent")
    size: Vec4,

    @doc("Particle orientation in the last frame")
    orientation: Quat,

    @doc("Particle orientation in the current frame")
    predicted_orientation: Quat,

    @doc("Particle density")
    density: float,

    @map("Vector<index_t>")
    edges: Map,

    bind_pose_center_of_mass: Vec4,
    bind_pose_inverse_bind_pose: Mat4,

    center_of_mass: Vec4,
    goal_position: Vec4,
    internal_forces: Vec4,

    @uniform
    @nonattribute
    global_center_of_mass: Vec4,

    @uniform
    @nonattribute
    light_source_direction: Vec4,

    @nonattribute
    colliders_sdf: SDF_Slot,

    @nonattribute
    colliders_mesh: Mesh_Collider_Slot,

    @nonattribute
    collision_constraints: Collision_Constraint,

    @set(index_t)
    @nonattribute
    fixed_particles: Set
};
