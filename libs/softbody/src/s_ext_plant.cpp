// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: plant simulation
//

#include "stdafx.h"
#include <iterator>
#include "system_state.h"
#include "softbody.h"
#include "s_ext.h"
#include "m_utils.h"
#include "l_random.h"
#include "s_iterators.h"
#include "f_serialization.internal.h"
#include "s_compute_backend.h"
#include <raymarching.h>
#include <intersect.h>

// TODO(danielm): duplicate of the implementation in s_compute_ref.cpp!!!
static std::array<uint64_t, 3> get_vertex_indices(Mesh_Collider_Slot const &c, size_t triangle_index) {
    auto base = triangle_index * 3;
    return {
        c.vertex_indices[base + 0],
        c.vertex_indices[base + 1],
        c.vertex_indices[base + 2]
    };
}

static std::array<uint64_t, 3> get_normal_indices(Mesh_Collider_Slot const &c, size_t triangle_index) {
    auto base = triangle_index * 3;
    return {
        c.normal_indices[base + 0],
        c.normal_indices[base + 1],
        c.normal_indices[base + 2]
    };
}

static std::function<float(Vec3 const&)> make_sdf_ast_wrapper(
        sb::sdf::ast::Expression<float>* expr,
        sb::sdf::ast::Sample_Point* sp) {
    return [expr, sp](Vec3 const& p) -> float {
        assert(expr != NULL);
        assert(sp != NULL);
        sp->set_value(p);
        return expr->evaluate();
    };
}

class Plant_Simulation : public ISimulation_Extension, public sb::IPlant_Simulation {
public:
    Plant_Simulation(sb::Config const &params, ICompute_Backend *compute)
        : _params(params)
        , _compute(compute) {
        if (params.extra.plant_sim != nullptr) {
            _extra = *params.extra.plant_sim;
        } else {
            // NOTE(danielm): plant_sim will be null when we're deserializing
            // a simulation image (we don't save ext params).
            // TODO(danielm): fix this
            _extra = {};
        }
    }
private:
    sb::Config _params;
    ICompute_Backend *_compute;
    sb::Plant_Simulation_Extension_Extra _extra;
    Rand_Float _rnd;
    Map<index_t, index_t> _parents;
    Map<index_t, index_t> _apical_child;
    Map<index_t, index_t> _lateral_bud;
    Map<index_t, Vec4> _anchor_points;
    Map<index_t, float> _lifetime;
    Dequeue<index_t> _growing;
    Vector<index_t> _leaf_bud;

    void init(IParticle_Manager_Deferred* pman, System_State& s, float dt) override {
        // Create root
        pman->defer([&](IParticle_Manager* pman, System_State&) {
            constexpr auto new_size = Vec3(0.5f, 0.5f, 2.0f);
            auto o = _extra.seed_position;
            auto root0 = pman->add_init_particle(o - Vec3(0, 0.5, 0), new_size, 1);
            auto root1 = pman->add_init_particle(o, new_size, 1);
            pman->connect_particles(root0, root1);

            index_t indices[] = { root0, root1 };

            pman->add_fixed_constraint(2, indices);
        });
    }


    void post_prediction(IParticle_Manager_Deferred* pman, System_State& s, float dt) override {
        auto const surface_adaption_min_dist = 8.0f;
        auto const attachment_min_dist = 0.05f;

        // For all particles
        for (index_t i = 0; i < s.position.size(); i++) {
            auto const p = s.predicted_position[i];
            if (_anchor_points.count(i)) {
                auto ap = _anchor_points[i];
                auto dist = distance(p, ap);
                if (dist <= attachment_min_dist) {
                    s.predicted_position[i] += dt * _extra.attachment_strength * (ap - s.predicted_position[i]);
                } else {
                    _anchor_points.erase(i);
                }
            } else {
                SDF_Slot const* surface = NULL;
                float surface_dist = INFINITY;

                // Find the closest surface
                for (auto& C : s.colliders_sdf) {
                    if(C.used) {
                        auto l = make_sdf_ast_wrapper(C.expr, C.sp);
                        auto dist = l(p);
                        if (dist < surface_dist) {
                            surface = &C;
                            surface_dist = dist;
                        }
                    }
                }

                for (auto &C : s.colliders_mesh) {
                }

                // If the surface is close enough, move towards it
                if (surface != NULL && surface_dist < surface_adaption_min_dist) {
                    auto surface_fun = make_sdf_ast_wrapper(surface->expr, surface->sp);
                    auto normal = sdf::normal(surface_fun, p);
                    auto surface = p - surface_dist * normal;

                    s.predicted_position[i] += dt * _extra.surface_adaption_strength * (surface - s.predicted_position[i]);

                    if (surface_dist < attachment_min_dist) {
                        _anchor_points[i] = surface;
                    }
                } else {
                    // No anchor point found, move towards light source
                    s.predicted_position[i] += dt * _extra.phototropism_response_strength * s.light_source_direction;
                }
            }

            if (_lifetime.count(i) != 0) {
                _lifetime[i] += dt;
            } else {
                _lifetime[i] = dt;
            }

            if (_lifetime[i] > 3) {
                s.fixed_particles.insert(i);
            }
        }
    }

    void post_integration(IParticle_Manager_Deferred* pman_defer, System_State& s, float dt) override {
        growth(pman_defer, s, dt);
    }

    void
    try_grow_branches(
        Vector<index_t> const &particleIndices,
        Vector<index_t> &couldntGrow,
        IParticle_Manager_Deferred *pman_defer,
        System_State &s,
        float dt) {
        constexpr auto new_size = Vec3(0.25f, 0.25f, 2.0f);

        struct Lateral_Branch_To_Grow {
            index_t pidx;
            Vec4 branch_dir;
            float branch_len;
        };

        struct Apical_Branch_To_Grow {
            index_t pidx;
            Vec4 branch_dir;
            float branch_len;
        };

        std::vector<Lateral_Branch_To_Grow> lateral_branches_to_grow;
        std::vector<Apical_Branch_To_Grow> apical_branches_to_grow;

        // - Filter out particles that won't grow a branch
        // - Gather information about particles that will grow lateral branches
        // - Gather information about particles that will grow apical branches
        for (index_t i = 0; i < particleIndices.size(); i++) {
            auto pidx = particleIndices[i];
            // Don't grow branches out of particles that are too fast
            if (length(s.velocity[pidx]) > 2) {
                couldntGrow.emplace_back(pidx);
                continue;
            }

            auto lateral_chance = _rnd.normal();

            // Compute the direction of the apical branch
            auto parent = _parents[pidx];
            auto branch_dir
                = normalize(s.bind_pose[pidx] - s.bind_pose[parent]);
            auto branch_len = 1.0f;

            apical_branches_to_grow.push_back({ pidx, branch_dir, branch_len });

            if (lateral_chance < _extra.branching_probability) {
                lateral_branches_to_grow.push_back(
                    { pidx, branch_dir, branch_len });
            } else {
                // This particle won't grow a lateral branch
                // We're adding this particle to the list of leaf buds
                _leaf_bud.emplace_back(pidx);
            }
        }

        // Gather all information needed to check whether the branch we want to
        // create would intersect the world geometry or not.
        //
        // We're doing the intersection tests in two batches because we'll use
        // the results in two different ways depending on what kind of branch
        // we're talking about.

        struct Intersection_Test_Batch {
            Vector<index_t> pidx;
            Vector<unsigned> results;
            Vector<Vec3> from;
            Vector<Vec3> to;
        };

        Intersection_Test_Batch lateral_branch_intersection_test,
            apical_branch_intersection_test;

        for (auto &B : lateral_branches_to_grow) {
            // Compute the direction of the lateral branch and the endpoint of
            // the branch

            auto angle = _rnd.central() * _extra.branch_angle_variance;
            auto x = _rnd.central();
            auto y = _rnd.central();
            auto z = _rnd.central();
            auto axis = normalize(Vec3(x, y, z));
            auto bud_rot_offset = angleAxis(angle, axis);
            auto lateral_branch_dir = bud_rot_offset * B.branch_dir * conjugate(bud_rot_offset);

            auto pos = s.position[B.pidx] + B.branch_len * lateral_branch_dir;

            lateral_branch_intersection_test.pidx.emplace_back(B.pidx);
            lateral_branch_intersection_test.from.emplace_back(
                s.position[B.pidx]);
            lateral_branch_intersection_test.to.emplace_back(pos);
            lateral_branch_intersection_test.results.emplace_back(0);
        }

        for (auto& B : apical_branches_to_grow) {
            auto pos = s.position[B.pidx] + B.branch_len * B.branch_dir;

            apical_branch_intersection_test.pidx.emplace_back(B.pidx);
            apical_branch_intersection_test.from.emplace_back(
                s.position[B.pidx]);
            apical_branch_intersection_test.to.emplace_back(pos);
            apical_branch_intersection_test.results.emplace_back(0);
        }

        _compute->check_intersections(
            s, lateral_branch_intersection_test.results,
            lateral_branch_intersection_test.from,
            lateral_branch_intersection_test.to);

        _compute->check_intersections(
            s, apical_branch_intersection_test.results,
            apical_branch_intersection_test.from,
            apical_branch_intersection_test.to);

        // Check the results of the intersection tests for the apical branches
        for (index_t i = 0; i < apical_branch_intersection_test.results.size();
             i++) {
            if (!apical_branch_intersection_test.results[i]) {
                auto pos = apical_branch_intersection_test.to[i];
                auto pidx = apical_branch_intersection_test.pidx[i];

                pman_defer->defer([&, pos, new_size, pidx](
                                      IParticle_Manager *pman,
                                      System_State &s) {
                    auto a_idx = pman->add_particle(pos, new_size, 1.0f, pidx);
                    _apical_child[pidx] = a_idx;
                    _parents[a_idx] = pidx;
                });
            }
        }

        // Check the results of the intersection tests for the lateral branches
        for (index_t i = 0; i < lateral_branch_intersection_test.results.size();
             i++) {
            if (!lateral_branch_intersection_test.results[i]) {
                auto pos = lateral_branch_intersection_test.to[i];
                auto pidx = lateral_branch_intersection_test.pidx[i];

                pman_defer->defer([&, pos, new_size, pidx](
                                      IParticle_Manager *pman,
                                      System_State &s) {
                    auto l_idx = pman->add_particle(pos, new_size, 1.0f, pidx);
                    _lateral_bud[pidx] = l_idx;
                    _parents[l_idx] = pidx;
                });
            }
        }
    }

    bool checkIntersection(System_State const &s, Vec3 from, Vec3 to) {
        for (auto const &coll : s.colliders_mesh) {
            if (!coll.used)
                continue;

            auto const dir = to - from;

            // for every triangle in coll: check intersection

            // TODO(danielm): check each triangle but do a minimum search by
            // `t` so that we only consider the nearest intersected surf?
            // cuz rn this may create multiple collision constraints for a
            // particle
            for (auto j = 0ull; j < coll.triangle_count; j++) {
                auto base = j * 3;
                glm::vec3 xp;
                float t;
                // TODO(danielm): these matrix vector products could be cached
                auto [vi0, vi1, vi2] = get_vertex_indices(coll, j);
                auto [ni0, ni1, ni2] = get_normal_indices(coll, j);
                auto v0 = coll.transform * Vec4(coll.vertices[vi0], 1);
                auto v1 = coll.transform * Vec4(coll.vertices[vi1], 1);
                auto v2 = coll.transform * Vec4(coll.vertices[vi2], 1);
                if (intersect::ray_triangle(xp, t, from, dir, v0, v1, v2) || t > 1) {
                    return true;
                }
            }
        }

        return false;
    }

    void growth(IParticle_Manager_Deferred* pman_defer, System_State& s, float dt) {
        auto g = 1.0f; // 1/sec
        auto const N = s.position.size();
        auto const max_size = 1.5f;

        if (N >= _extra.particle_count_limit) return;

        for (index_t pidx = 1; pidx < N; pidx++) {
            auto r = s.size[pidx].x;

            s.size[pidx] += Vec4(g * dt, g * dt, 0, 0);
            if (r < max_size) {
                r = s.size[pidx].x;

                // Amint tullepjuk a reszecske meret limitet, novesszunk uj agat
                if (r >= max_size && N < _extra.particle_count_limit) {
                    // don't grow if particle is underground
                    // TODO(danielm): here we assume that Y=0 is the ground plane
                    if (s.position[pidx].y >= 0) {
                        _growing.push_back(pidx);
                    }

                    s.bind_pose[pidx] = s.position[pidx];
                    // TODO(danielm): don't we need to recalculate some
                    // matrices here?
                }
            }
        }

        // Move the contents of _growing into a vector
        auto growing = std::vector<index_t>();
        std::move(begin(_growing), end(_growing), back_inserter(growing));
        _growing.clear();
        Vector<index_t> couldntGrow;

        try_grow_branches(growing, couldntGrow, pman_defer, s, dt);

        std::move(
            begin(couldntGrow), end(couldntGrow), back_inserter(_growing));
    }

    sb::Unique_Ptr<sb::Relation_Iterator> get_parental_relations() override {
        auto get_map = [&]() -> decltype(_parents)& { return _parents; };
        auto make_relation = [&](index_t child, index_t parent) {
            return sb::Relation {
                parent,
                Vec4(),
                child,
                Vec4(),
            };
        };

        return std::make_unique<One_To_One_Relation_Iterator>(get_map, make_relation);
    }

    std::vector<index_t>
    get_leaf_buds() override {
        return _leaf_bud;
    }

#define MAX_POSSIBLE_CHUNK_COUNT (5)
#define VERSION (0x00000000)
#define CHUNK_PARENTS       MAKE_4BYTE_ID('P', 'S', 'p', 'a')
#define CHUNK_APICAL_CHILD  MAKE_4BYTE_ID('P', 'S', 'a', 'c')
#define CHUNK_LATERAL_BUD   MAKE_4BYTE_ID('P', 'S', 'l', 'b')
#define CHUNK_ANCHOR_POINTS MAKE_4BYTE_ID('P', 'S', 'a', 'p')
#define CHUNK_LEAF_BUD      MAKE_4BYTE_ID('P', 'S', 'f', 'b')

    bool save_image(sb::ISerializer* serializer, System_State const& s) override {
        uint32_t id = Extension_Lookup_Chunk_Identifier(sb::Extension::Plant_Simulation);
        uint32_t sentinel = MAKE_4BYTE_ID('P', 'S', 's', 'n');
        uint32_t version = VERSION;
        uint32_t chunk_count = MAX_POSSIBLE_CHUNK_COUNT;

        serializer->write(&id, sizeof(id));
        serializer->write(&sentinel, sizeof(sentinel));
        serializer->write(&version, sizeof(version));
        serializer->write(&chunk_count, sizeof(chunk_count));

        serialize(serializer, _parents, CHUNK_PARENTS);
        serialize(serializer, _apical_child, CHUNK_APICAL_CHILD);
        serialize(serializer, _lateral_bud, CHUNK_LATERAL_BUD);
        serialize(serializer, _anchor_points, CHUNK_ANCHOR_POINTS);
        serialize(serializer, _leaf_bud, CHUNK_LEAF_BUD);

        return true;
    }

    bool load_image(sb::IDeserializer* deserializer, System_State& s) override {
        uint32_t sentinel, chunk_count, version;

        deserializer->read(&sentinel, sizeof(sentinel));
        if (sentinel != MAKE_4BYTE_ID('P', 'S', 's', 'n')) {
            return false;
        }

        deserializer->read(&version, sizeof(version));
        if (version > VERSION) {
            return false;
        }

        deserializer->read(&chunk_count, sizeof(chunk_count));
        if (chunk_count <= MAX_POSSIBLE_CHUNK_COUNT) {
            uint32_t id;

            for (uint32_t chunk = 0; chunk < chunk_count; chunk++) {
                deserializer->read(&id, sizeof(id));

                switch (id) {
                case CHUNK_PARENTS: deserialize(deserializer, _parents); break;
                case CHUNK_APICAL_CHILD: deserialize(deserializer, _apical_child); break;
                case CHUNK_LATERAL_BUD: deserialize(deserializer, _lateral_bud); break;
                case CHUNK_ANCHOR_POINTS: deserialize(deserializer, _anchor_points); break;
                case CHUNK_LEAF_BUD: deserialize(deserializer, _leaf_bud); break;
                default: printf("UNKNOWN CHUNK ID %x\n", id); std::terminate(); break;
                }
            }
        } else {
            return false;
        }

        return true;
    }

    bool wants_to_serialize() override {
        return true;
    }
};

sb::Unique_Ptr<ISimulation_Extension> Create_Extension_Plant_Simulation(sb::Extension kind, sb::Config const& params, ICompute_Backend *compute) {
    return std::make_unique<Plant_Simulation>(params, compute);
}
