// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: softbody simulation
//

#include "stdafx.h"
#include "softbody.h"
#include "m_utils.h"
#include "l_iterators.h"
#include "s_simulation.h"
#include "logger.h"
#include "collider_handles.h"
#include <cstdlib>
#include <cstdarg>
#include <array>
#include <raymarching.h>
#include <glm/gtx/matrix_operation.hpp>
#include "s_compute_backend.h"
#include "f_serialization.h"

#include <Tracy.hpp>

#define PHYSICS_STEP (1.0f / 25.0f)
#define TAU (PHYSICS_STEP)
#define SIM_SIZE_LIMIT (4096)
#define SOLVER_ITERATIONS (12)

// #define DEBUG_TETRAHEDRON
#ifdef DEBUG_TETRAHEDRON
#define DISABLE_GROWTH
#define DISABLE_PHOTOTROPISM
#endif /* defined(DEBUG_TETRAHEDRON) */

// #define DISABLE_GROWTH
// #define DISABLE_PHOTOTROPISM

#define DEFER_LAMBDA(lambda)                \
{                                           \
    Lock_Guard g(deferred_lock);            \
    deferred.push_back(std::move(lambda));   \
}

#if NDEBUG
#define ON_DEBUG(expr)
#else
#define ON_DEBUG(expr) expr
#endif

template<unsigned Order, typename T>
struct Cache_Table {
    constexpr static unsigned Size = 1 << Order;
    constexpr static unsigned Mask = Size - 1;
    struct Key_Value {
        unsigned k;
        T value;
    };

    Key_Value table[Size];

    Cache_Table() {
        clear();
    }

    void clear() {
        for (auto& kv : table) {
            kv.k = UINT_MAX;
            kv.value = T();
        }
    }

    T fetch_or_insert(unsigned key, std::function<T(unsigned key)> f) {
        auto off = key & Mask;
        assert(off < Size);

        auto& slot = table[off];
        if (slot.k == key) {
            return slot.value;
        }

        slot.k = key;
        slot.value = f(key);
        return slot.value;
    }
};

Softbody_Simulation::Softbody_Simulation(sb::Config const& configuration, sb::Debug_Proc dbg_msg_cb, void* dbg_msg_user)
    : assert_parallel(false), assert_init(true), debugproc(dbg_msg_cb), debugproc_user(dbg_msg_user) {
    auto o = glm::vec3(0, 0, 0);
#ifdef DEBUG_TETRAHEDRON
#if 1
    auto siz = Vec3(1, 1, 2);
    auto idx_root = add_init_particle(o, siz, 1);
    auto idx_t0 = add_init_particle(o + Vec3(-4,  8,  4), siz, 1);
    auto idx_t1 = add_init_particle(o + Vec3(+4,  8,  4), siz, 1);
    auto idx_t2 = add_init_particle(o + Vec3( 0,  8, -4), siz, 1);

    s.fixed_particles.insert(idx_root);

    connect_particles(idx_root, idx_t0);
    connect_particles(idx_root, idx_t1);
    connect_particles(idx_root, idx_t2);
    connect_particles(idx_t0, idx_t1);
    connect_particles(idx_t1, idx_t2);
    connect_particles(idx_t0, idx_t2);
#else
    auto siz = Vec3(0.25, 0.25, 0.5);
    auto x90 = glm::normalize(Quat(Vec3(0, 0, glm::radians(90.0f))));
    auto idx_root = add_particle(o, siz, 1);
    s.orientation[idx_root] = x90;
    auto prev = idx_root;
    for (int i = 0; i < 64; i++) {
        auto cur = add_init_particle(o + Vec3((i + 1) * 0.5f, 0, 0), siz, 1);
        s.orientation[cur] = x90;
        connect_particles(prev, cur);
        prev = cur;
    }
#endif
#else
#endif /* DEBUG_TETRAHEDRON */

    params = configuration;

    s.light_source_direction = Vec4(0, 1, 0, 0);

    s.center_of_mass.resize(particle_count());

    compute = Make_Compute_Backend(configuration.compute_preference, this);
    create_extension(params.ext, params);
    params.extra.ptr = NULL;
    pump_deferred_requests();
}

void Softbody_Simulation::prediction(float dt) {
    assert_init = false;
    s.predicted_position.resize(particle_count());
    s.predicted_orientation.resize(s.orientation.size());
    s.center_of_mass.resize(particle_count());

    ext->pre_prediction(this, s, dt);

    compute->predict(s, dt);

    compute->generate_collision_constraints(s);

    ext->post_prediction(this, s, dt);
}

#define NUMBER_OF_CLUSTERS(idx) (s.edges[(idx)].size() + 1)

void Softbody_Simulation::pump_deferred_requests() {
    for (auto& def_func : deferred) {
        def_func(this, s);
    }
    deferred.clear();
}

float Softbody_Simulation::get_phdt() {
    return PHYSICS_STEP;
}

void Softbody_Simulation::do_one_iteration_of_distance_constraint_resolution(float phdt) {
    compute->do_one_iteration_of_distance_constraint_resolution(s, phdt);
}

void Softbody_Simulation::do_one_iteration_of_fixed_constraint_resolution(float phdt) {
    // force particles to stay in their bind pose
    compute->do_one_iteration_of_fixed_constraint_resolution(s, phdt);
}

void Softbody_Simulation::constraint_resolution(float dt) {
    ext->pre_constraint(this, s, dt);

    for (auto iter = 0ul; iter < SOLVER_ITERATIONS; iter++) {
        do_one_iteration_of_distance_constraint_resolution(dt);
        do_one_iteration_of_fixed_constraint_resolution(dt);
        do_one_iteration_of_collision_constraint_resolution(dt);
    }

    ext->post_constraint(this, s, dt);
}

void Softbody_Simulation::do_one_iteration_of_collision_constraint_resolution(float phdt) {
    compute->do_one_iteration_of_collision_constraint_resolution(s, phdt);
}

void Softbody_Simulation::integration(float dt) {
    ext->pre_integration(this, s, dt);
    compute->integrate(s, dt);
    ext->post_integration(this, s, dt);
}

bool Softbody_Simulation::add_collider(
        sb::ISoftbody_Simulation::Collider_Handle& out_handle,
        sb::sdf::ast::Expression<float>* expr,
        sb::sdf::ast::Sample_Point* sp) {

    if(expr == NULL || sp == NULL) {
        return false;
    }

    SDF_Slot* slot = NULL;
    for (auto i = 0ull; i < s.colliders_sdf.size(); i++) {
        if (!s.colliders_sdf[i].used) {
            slot = &s.colliders_sdf[i];
            out_handle = make_collider_handle(Collider_Handle_Kind::SDF, i);
            break;
        }
    }

    if (slot == NULL) {
        out_handle = make_collider_handle(Collider_Handle_Kind::SDF, s.colliders_sdf.size());
        s.colliders_sdf.push_back({});
        slot = &s.colliders_sdf.back();
    }

    slot->used = true;
    slot->expr = expr;
    slot->sp = sp;

    compute->on_collider_added(s, out_handle);

    return true;
}

bool Softbody_Simulation::remove_collider(Collider_Handle h) {
    Collider_Handle_Kind kind;
    size_t index;
    decode_collider_handle(h, kind, index);

    compute->on_collider_removed(s, h);

    switch (kind) {
    case Collider_Handle_Kind::SDF:
    {
        if (index < s.colliders_sdf.size()) {
            s.colliders_sdf[index].used = false;
            s.colliders_sdf[index].expr = NULL;
            s.colliders_sdf[index].sp = NULL;
            return true;
        } else {
            return false;
        }
        break;
    }
    case Collider_Handle_Kind::Mesh:
    {
        if (index < s.colliders_mesh.size()) {
            s.colliders_mesh[index].used = false;
            return true;
        } else {
            return false;
        }
        break;
    }
    default:
        assert(!"Unhandled collider handle kind!");
        break;
    }

    return false;
}

void Softbody_Simulation::collider_changed(Collider_Handle h) {
    Collider_Handle_Kind kind;
    size_t index;
    decode_collider_handle(h, kind, index);

    if (kind == Collider_Handle_Kind::SDF) {
        if (index < s.colliders_sdf.size() || s.colliders_sdf[index].used) {
            compute->on_collider_changed(s, h);
        }
    }
}

bool Softbody_Simulation::add_collider(Collider_Handle &out_handle, sb::Mesh_Collider const *mesh) {
    assert(mesh != nullptr);
    if (mesh == nullptr ||
        mesh->indices == nullptr ||
        mesh->positions == nullptr ||
        mesh->num_positions == 0 ||
        mesh->normals == nullptr ||
        mesh->num_normals == 0 ||
        mesh->triangle_count == 0) {
        return false;
    }

    // Allocate slot
    Mesh_Collider_Slot* slot = nullptr;
    // Look for an unused slot
    for (auto i = 0ull; i < s.colliders_mesh.size(); i++) {
        if (!s.colliders_mesh[i].used) {
            slot = &s.colliders_mesh[i];
            out_handle = make_collider_handle(Collider_Handle_Kind::Mesh, i);
            break;
        }
    }

    if (slot == NULL) {
        // Create new slot
        out_handle = make_collider_handle(Collider_Handle_Kind::Mesh, s.colliders_mesh.size());
        s.colliders_mesh.push_back({});
        slot = &s.colliders_mesh.back();
    }

    slot->used = true;
    slot->transform = mesh->transform;
    slot->triangle_count = mesh->triangle_count;

    slot->vertex_indices.reserve(mesh->triangle_count * 3);
    slot->normal_indices.reserve(mesh->triangle_count * 3);

    for (auto i = 0ull; i < mesh->triangle_count; i++) {
        slot->vertex_indices.push_back(mesh->indices[i * 3 + 0]);
        slot->vertex_indices.push_back(mesh->indices[i * 3 + 1]);
        slot->vertex_indices.push_back(mesh->indices[i * 3 + 2]);

        slot->normal_indices.push_back(mesh->indices[i * 3 + 0]);
        slot->normal_indices.push_back(mesh->indices[i * 3 + 1]);
        slot->normal_indices.push_back(mesh->indices[i * 3 + 2]);
    }

    slot->vertices.reserve(mesh->num_positions);
    slot->normals.reserve(mesh->num_normals);

    for (auto i = 0ull; i < mesh->num_positions; i++) {
        auto b = i * 3; // base index
        slot->vertices.push_back(Vec3(mesh->positions[b + 0], mesh->positions[b + 1], mesh->positions[b + 2]));
    }

    for (auto i = 0ull; i < mesh->num_normals; i++) {
        auto b = i * 3; // base index
        slot->normals.push_back(Vec3(mesh->normals[b + 0], mesh->normals[b + 1], mesh->normals[b + 2]));
    }

    compute->on_collider_added(s, out_handle);

    return true;
}

bool Softbody_Simulation::update_transform(Collider_Handle handle, glm::mat4 const &transform) {
    Collider_Handle_Kind kind;
    size_t index;
    decode_collider_handle(handle, kind, index);

    if (kind != Collider_Handle_Kind::Mesh) {
        return false;
    }

    if (index >= s.colliders_mesh.size()) {
        return false;
    }

    if (!s.colliders_mesh[index].used) {
        return false;
    }

    s.colliders_mesh[index].transform = transform;
    collider_changed(handle);
    return true;
}

index_t Softbody_Simulation::add_init_particle(Vec3 const& p_pos, Vec3 const& p_size, float p_density) {
    assert(!assert_parallel);
    assert(p_density >= 0.0f && p_density <= 1.0f);
    index_t const index = create_element(s);
    auto pos = Vec4(p_pos, 0);
    auto size = Vec4(p_size, 0);
    s.bind_pose[index] = pos;
    s.position[index] = pos;
    s.predicted_position[index] = pos;
    s.goal_position[index] = pos;
    s.size[index] = size;
    s.density[index] = p_density;
    s.edges[index] = {};

    invalidate_particle_cache(index);

    return index;
}

void Softbody_Simulation::connect_particles(index_t a, index_t b) {
    assert(!assert_parallel);
    assert(a != b);
    assert(a < particle_count());
    assert(b < particle_count());

    s.edges[a].push_back(b);
    s.edges[b].push_back(a);

    invalidate_particle_cache(a);
    invalidate_particle_cache(b);
}

index_t Softbody_Simulation::add_particle(Vec3 const& p_pos, Vec3 const& p_size, float p_density, index_t parent) {
    assert(!assert_parallel);
    assert(p_density >= 0.0f && p_density <= 1.0f);

    index_t const index = create_element(s);
    assert(parent < index);

    auto pos = Vec4(p_pos, 0);
    auto size = Vec4(p_size, 0);

    // Determine the bind pose position
    auto parent_pos = s.position[parent];
    auto offset = pos - parent_pos;
    auto p_bpos = s.bind_pose[parent] + offset;

    s.bind_pose[index] = p_bpos;
    s.position[index] = pos;
    s.predicted_position[index] = pos;
    s.goal_position[index] = pos;
    s.size[index] = size;
    s.density[index] = p_density;
    s.edges[index] = {};

// #define INHERIT_MOMENTUM
#ifdef INHERIT_MOMENTUM
    auto parent_momentum = mass_of_particle(parent) * s.velocity[parent];
    auto child_velocity = parent_momentum / mass_of_particle(index);
    s.velocity[index] = child_velocity;
#endif

    // NOTE(danielm): connect_particle invalidates all cached info about both
    // this particle and its parent
    connect_particles(index, parent);

    return index;
}

float Softbody_Simulation::mass_of_particle(index_t i) {
    auto const d_i = s.density[i];
    auto const s_i = s.size[i];
    auto const m_i = (4.f / 3.f) * glm::pi<float>() * s_i.x * s_i.y * s_i.z * d_i;
    return m_i;
}

void Softbody_Simulation::invalidate_particle_cache() {
    auto N = particle_count();
    for (index_t i = 0; i < N; i++) {
        invalidate_particle_cache(i);
    }
}

sb::IPlant_Simulation* Softbody_Simulation::get_extension_plant_simulation() {
    if (params.ext == sb::Extension::Plant_Simulation) {
        // TODO(danielm): we should be storing the extension in a tagged union,
        // so we dont have to downcast
        return dynamic_cast<sb::IPlant_Simulation*>(ext.get());
    } else {
        return nullptr;
    }
}

void Softbody_Simulation::invalidate_particle_cache(index_t pidx) {
    auto& neighbors = s.edges[pidx];

    auto M = std::accumulate(
        neighbors.begin(), neighbors.end(),
        mass_of_particle(pidx),
        [&](float acc, index_t idx) {
            return acc + mass_of_particle(idx);
        }
    );

    auto com0 = std::accumulate(
        neighbors.begin(), neighbors.end(),
        mass_of_particle(pidx) * s.bind_pose[pidx],
        [&](Vec4 const& acc, index_t idx) {
            return acc + mass_of_particle(idx) * s.bind_pose[idx];
        }
    ) / M;

    auto calc_A_0_i = [&](index_t i) -> Mat3 {
        auto q_i = s.bind_pose[i] - com0;
        auto m_i = mass_of_particle(i);

        return m_i * glm::outerProduct(q_i, q_i);
    };

    // A_qq
    Mat3 A_0 = std::accumulate(
        neighbors.begin(), neighbors.end(),
        calc_A_0_i(pidx),
        [&](auto acc, auto idx) { return acc + calc_A_0_i(idx); }
    );

    Mat3 invRest;

    if (glm::abs(glm::determinant(A_0)) > glm::epsilon<float>()) {
        invRest = glm::inverse(A_0);
    } else {
        invRest = Mat3(1.0f);
    }

    size_t i = pidx;
    if (s.bind_pose_center_of_mass.size() <= i) {
        s.bind_pose_center_of_mass.resize(i + 1);
    }

    if (s.bind_pose_inverse_bind_pose.size() <= i) {
        s.bind_pose_inverse_bind_pose.resize(i + 1);
    }

    s.bind_pose_center_of_mass[i] = com0;
    s.bind_pose_inverse_bind_pose[i] = invRest;
}

void Softbody_Simulation::defer(std::function<void(IParticle_Manager* pman, System_State& s)> const& f) {
    std::lock_guard G(deferred_lock);
    deferred.push_back(f);
}

bool Softbody_Simulation::save_image(sb::ISerializer* serializer) {
    return sim_save_image(s, serializer, ext.get());
}

bool Softbody_Simulation::load_image(sb::IDeserializer* deserializer) {
    auto res = sim_load_image(this, s, deserializer, params);

    return res == Serialization_Result::OK;
}

void Softbody_Simulation::add_particles(int N, glm::vec4 const* positions) {
    for (int i = 0; i < N; i++) {
        add_init_particle(positions[i], Vec4(1, 1, 1, 0), 1);
    }
}

void Softbody_Simulation::add_connections(int N, long long* pairs) {
    auto pn = s.position.size();
    for (int i = 0; i < N; i++) {
        auto i0 = pairs[i * 2 + 0];
        auto i1 = pairs[i * 2 + 1];

        assert(i0 < pn);
        assert(i1 < pn);

        if (i0 >= pn || i1 >= pn) {
            continue;
        }

        assert(s.edges.count(i0));
        auto& n0 = s.edges[i0];
        auto fit0 = std::find(n0.cbegin(), n0.cend(), i1);
        if (fit0 == n0.cend()) {
            n0.push_back(i1);
        }

        assert(s.edges.count(i1));
        auto& n1 = s.edges[i1];
        auto fit1 = std::find(n1.cbegin(), n1.cend(), i0);
        if (fit1 == n1.cend()) {
            n1.push_back(i0);
        }
    }
}

ISimulation_Extension* Softbody_Simulation::create_extension(sb::Extension kind, sb::Config const& config) {
    assert(compute != nullptr);
    ext = Create_Extension(kind, config, compute.get());
    params = config;
    params.ext = kind;
    ext->init(this, s, 0.0f);

    return ext.get();
}

void Softbody_Simulation::add_fixed_constraint(unsigned count, index_t* pidx) {
    assert(!assert_parallel);
    assert(pidx != NULL);

    if (count > 0 && pidx != NULL) {
        s.fixed_particles.insert(pidx, pidx + count);
    }
}

sb::Unique_Ptr<sb::ISoftbody_Simulation> sb::create_simulation(Config const& configuration, sb::Debug_Proc dbg_msg_cb, void* dbg_msg_user) {
    return std::make_unique<Softbody_Simulation>(configuration, dbg_msg_cb, dbg_msg_user);
}

void Softbody_Simulation::set_light_source_position(Vec3 const& pos) {
    s.light_source_direction = Vec4(pos, 0);
}

void Softbody_Simulation::step(float delta_time) {
    ZoneScoped;

    if(s.position.size() == 0) {
        return;
    }

    time_accumulator += delta_time;

    if (time_accumulator > PHYSICS_STEP) {
        // FrameMarkStart("Softbody");
        auto phdt = PHYSICS_STEP;

        compute->begin_new_frame(s);
        prediction(phdt);
        constraint_resolution(phdt);
        compute->dampen(s, phdt);

        compute->generate_collision_constraints(s);
        for (int i = 0; i < 4; i++) {
            compute->do_one_iteration_of_collision_constraint_resolution(s, phdt);
        }
        compute->do_one_iteration_of_fixed_constraint_resolution(s, phdt);

        integration(phdt);
        compute->end_frame(s);

        if (time_accumulator > 8 * PHYSICS_STEP) {
            log(sb::Debug_Message_Source::Simulation_Driver, sb::Debug_Message_Type::Debug, sb::Debug_Message_Severity::Low, "extreme-lag acc=%f", time_accumulator);
        }

        pump_deferred_requests();

        time_accumulator -= phdt;
        // FrameMarkEnd("Softbody");
    }
}

void Softbody_Simulation::set_debug_visualizer(sb::IDebug_Visualizer *pVisualizer) {
    m_pVisualizer = pVisualizer;
    
    compute->set_debug_visualizer(pVisualizer);
}

sb::Unique_Ptr<sb::ISingle_Step_State> Softbody_Simulation::begin_single_step() {
    return nullptr;
}

void Softbody_Simulation::debug_message_callback(sb::Debug_Proc callback, void* user) {
    debugproc = callback;
    debugproc_user = user;
}

void Softbody_Simulation::log(sb::Debug_Message_Source s, sb::Debug_Message_Type t, sb::Debug_Message_Severity l, char const* fmt, ...) {
    if(debugproc != nullptr) {
        int size = 128;
        char* buf;
        va_list ap;
        int n;

        // Does it fit into 128 bytes?
        buf = new char[size];
        va_start(ap, fmt);
        n = vsnprintf(buf, size, fmt, ap);
        va_end(ap);

        if(n >= size) {
            // It doesn't, so we resize the buffer to the exact amount needed
            size = n + 1;
            delete[] buf;
            buf = new char[size];

            va_start(ap, fmt);
            n = vsnprintf(buf, size, fmt, ap);
            va_end(ap);
        }

        debugproc(s, t, l, buf, debugproc_user);

        delete[] buf;
    }
}
