// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: reference computation backend
//

#include "stdafx.h"
#include <cassert>
#include <array>
#include <algorithm>
#include "softbody.h"
#include "l_iterators.h"
#include "s_compute_backend.h"
#define SB_BENCHMARK (1)
#define SB_BENCHMARK_UNITS microseconds
#define SB_BENCHMARK_UNITS_STR "us"
#include "s_benchmark.h"
#include "m_utils.h"
#include <glm/gtx/matrix_operation.hpp>

#define NUMBER_OF_CLUSTERS(idx) (s.edges[(idx)].size() + 1)

// TODO(danielm): duplicate of the implementation in objscan!!!
static bool intersect_ray_triangle(
    glm::vec3 &xp, float &t,
    glm::vec3 const &origin, glm::vec3 const &dir,
    glm::vec3 const &v0, glm::vec3 const &v1, glm::vec3 const &v2
) {
    auto edge1 = v2 - v0;
    auto edge0 = v1 - v0;
    auto h = cross(dir, edge1);
    auto a = dot(edge0, h);
    if (-glm::epsilon<float>() < a && a < glm::epsilon<float>()) {
        return false;
    }

    auto f = 1.0f / a;
    auto s = origin - v0;
    auto u = f * dot(s, h);

    if (u < 0 || 1 < u) {
        return false;
    }

    auto q = cross(s, edge0);
    auto v = f * dot(dir, q);

    if (v < 0 || u + v > 1) {
        return false;
    }

    t = f * dot(edge1, q);
    if (t <= glm::epsilon<float>()) {
        return false;
    }

    xp = origin + t * dir;
    return true;
}

static std::array<uint64_t, 3> get_vertex_indices(System_State::Mesh_Collider_Slot const &c, size_t triangle_index) {
    auto base = triangle_index * 3;
    return {
        c.vertex_indices[base + 0],
        c.vertex_indices[base + 1],
        c.vertex_indices[base + 2]
    };
}

static std::array<uint64_t, 3> get_normal_indices(System_State::Mesh_Collider_Slot const &c, size_t triangle_index) {
    auto base = triangle_index * 3;
    return {
        c.normal_indices[base + 0],
        c.normal_indices[base + 1],
        c.normal_indices[base + 2]
    };
}

#define CHECK_VISUALIZER_PRESENT_DEBUG(expr) \
    if(_visualizer != nullptr) expr;

class Compute_CPU_Single_Threaded : public ICompute_Backend {
public:
    Compute_CPU_Single_Threaded(ILogger* logger) : _log(logger) {
    }
protected:
    float mass_of_particle(System_State& s, index_t i) const {
        auto const d_i = s.density[i];
        auto const s_i = s.size[i];
        auto const m_i = (4.f / 3.f) * glm::pi<float>() * s_i.x * s_i.y * s_i.z * d_i;
        return m_i;
    }

    size_t particle_count(System_State& s) const {
        return s.position.size();
    }

    void begin_new_frame(System_State const& sim) override {
        CHECK_VISUALIZER_PRESENT_DEBUG(_visualizer->new_frame());
    }

    void set_debug_visualizer(sb::IDebug_Visualizer *visualizer) override {
        _visualizer = visualizer;
    }

    void velocity_damping(System_State& s, float dt) {
        auto N = particle_count(s);
        float d = 1 / glm::pow(2, dt);
        // TODO(danielm): this should have timestep-dependency
        for (index_t i = 0; i < N; i++) {
            s.velocity[i] *= d;
            s.angular_velocity[i] *= d;
        }

        return;

        // TODO(danielm): something is wrong with these calculations
        // Probably the r_i_tilde matrix

        auto k_damping = 0.5f;
        for (index_t i = 0; i < N; i++) {
            std::array<index_t, 1> me{ i };
            auto& neighbors = s.edges[i];
            auto neighbors_and_me = iterator_union(neighbors.begin(), neighbors.end(), me.begin(), me.end());
            auto x_cm = s.center_of_mass[i];

            auto M = std::accumulate(
                neighbors.begin(), neighbors.end(),
                mass_of_particle(s, i),
                [&](float acc, index_t pidx) {
                    return acc + mass_of_particle(s, pidx);
                }
            );

            auto v_cm = std::accumulate(
                neighbors.begin(), neighbors.end(),
                s.velocity[i] * mass_of_particle(s, i),
                [&](Vec4 const& acc, index_t pidx) {
                    return acc + s.velocity[pidx] * mass_of_particle(s, pidx);
                }
            ) / M;

            auto get_r_i = [&](index_t pidx) {
                return Vec3(s.position[pidx] - x_cm);
            };

            if (length(v_cm) > glm::epsilon<float>()) {
                auto L = std::accumulate(
                    neighbors.begin(), neighbors.end(),
                    cross(get_r_i(i), mass_of_particle(s, i) * Vec3(s.velocity[i])),
                    [&](Vec3 const& acc, index_t pidx) {
                        return acc + cross(get_r_i(i), mass_of_particle(s, i) * Vec3(s.velocity[i]));
                    }
                );

                auto get_r_i_tilde = [&](index_t pidx) -> Mat3 {
                    Mat3 ret(1.0f);
                    auto v = Vec3(s.velocity[pidx]);
                    if (length(v) > glm::epsilon<float>()) {
                        auto r_cross_v = cross(get_r_i(pidx), v);
                        ret[0][0] = r_cross_v.x / v.x;
                        ret[1][1] = r_cross_v.y / v.y;
                        ret[2][2] = r_cross_v.z / v.z;
                    }
                    return ret;
                };

                auto I = std::accumulate(
                    neighbors.begin(), neighbors.end(),
                    mass_of_particle(s, i) * get_r_i_tilde(i) * transpose(get_r_i_tilde(i)),
                    [&](Mat3 const& acc, index_t pidx) {
                        return mass_of_particle(s, pidx) * get_r_i_tilde(pidx) * transpose(get_r_i_tilde(pidx));
                    }
                );

                auto ang_v = glm::inverse(I) * L;

                for (auto pidx : neighbors_and_me) {
                    auto dv = v_cm + Vec4(cross(ang_v, get_r_i(pidx)), 0) - s.velocity[pidx];
                    s.velocity[i] += k_damping * dv;
                }
            }
        }
    }

    void predict(System_State& s, float dt) override {
        DECLARE_BENCHMARK_BLOCK();
        BEGIN_BENCHMARK();

        auto const N = particle_count(s);
        velocity_damping(s, dt);

        for (unsigned i = 0; i < N; i++) {
            // prediction step

            // TODO(danielm): sum forces acting on the system here instead of
            // hardcoding it to the gravitational force
            auto external_forces = Vec4(0, -10, 0, 0);
            auto v = s.velocity[i] + dt * (1 / mass_of_particle(s, i)) * external_forces;
            auto pos = s.position[i] + dt * v;

            auto ang_v = glm::length(s.angular_velocity[i]);
            Quat q;
            if (ang_v < 0.01) {
                // Angular velocity is too small; for stability reasons we keep
                // the old orientation
                q = s.orientation[i];
            } else {
                q = Quat(glm::cos(ang_v * dt / 2.0f), s.angular_velocity[i] / ang_v * glm::sin(ang_v * dt / 2.0f));
            }

            s.predicted_position[i] = pos;
            s.predicted_orientation[i] = q;
        }

        END_BENCHMARK();
        PRINT_BENCHMARK_RESULT(_log);
    }


    void integrate(System_State& s, float dt) override {
        auto const N = particle_count(s);

        for (unsigned i = 0; i < N; i++) {
            s.velocity[i] = (s.predicted_position[i] - s.position[i]) / dt;
            s.position[i] = s.predicted_position[i];

            auto r_tmp = s.predicted_orientation[i] * glm::conjugate(s.orientation[i]);
            auto r = (r_tmp.w < 0) ? -r_tmp : r_tmp;
            auto q_angle = glm::angle(r);
            if (glm::abs(q_angle) < 0.1) {
                s.angular_velocity[i] = Vec4(0, 0, 0, 0);
            } else {
                s.angular_velocity[i] = Vec4(glm::axis(r) * q_angle / dt, 0);
            }

            s.orientation[i] = s.predicted_orientation[i];
        }

        for (auto& C : s.collision_constraints) {
            s.velocity[C.pidx] = Vec4();
        }
    }

    void do_one_iteration_of_shape_matching_constraint_resolution(
            System_State& s,
            float phdt
            ) override {
        DECLARE_BENCHMARK_BLOCK();
        BEGIN_BENCHMARK();

        // shape matching constraint
        for (index_t i = 0; i < particle_count(s); i++) {
            std::array<index_t , 1> me{ i };
            auto& neighbors = s.edges[i];
            auto neighbors_and_me = iterator_union(neighbors.begin(), neighbors.end(), me.begin(), me.end());

            // Sum particle weights in the current cluster
            auto M = std::accumulate(
                    neighbors.begin(), neighbors.end(),
                    mass_of_particle(s, i),
                    [&](float acc, index_t idx) {
                    return acc + mass_of_particle(s, idx);
                    }
                    );

            assert(M != 0);

            auto invRest = s.bind_pose_inverse_bind_pose[i];
            auto com0 = s.bind_pose_center_of_mass[i];

            // Center of mass calculated using the predicted positions
            auto com_cur = std::accumulate(
                    neighbors.begin(), neighbors.end(),
                    mass_of_particle(s, i) * s.predicted_position[i],
                    [&](Vec4 const& acc, index_t idx) {
                    return acc + mass_of_particle(s, idx) * s.predicted_position[idx];
                    }
                    ) / M;

            s.center_of_mass[i] = com_cur;

            // Calculates the moment matrix of a single particle
            auto calc_A_i = [&](index_t i) -> Mat4 {
                auto m_i = mass_of_particle(s, i);
                auto A_i = 1.0f / 5.0f * glm::diagonal4x4(s.size[i] * s.size[i]) * Mat4(s.orientation[i]);

                return m_i * (A_i + glm::outerProduct(s.predicted_position[i], s.bind_pose[i]) - glm::outerProduct(com_cur, com0));
            };

            // Calculate the cluster moment matrix
            auto A = std::accumulate(
                neighbors.begin(), neighbors.end(),
                calc_A_i(i),
                [&](Mat4 const& acc, index_t idx) -> Mat4 {
                    return acc + calc_A_i(idx);
                }
            ) * invRest;

            // Extract the rotational part of A which is the least squares optimal
            // rotation that transforms the original bind-pose configuration into
            // the current configuration
            Quat R = s.predicted_orientation[i];
            mueller_rotation_extraction(A, R);

            float const stiffness = 1;

            // NOTE(danielm): not sure if we need to correct every particle in the
            // current cluster/group other than the group owner.
            // Works either way, but the commented out way would make parallelization
            // painful.
            // for (auto idx : neighbors_and_me) {
            {
                auto idx = i;
                // Bind pose position relative to the center of mass
                auto pos_bind = s.bind_pose[idx] - com0;
                // Current position relative to the center of mass
                auto d = s.predicted_position[idx] - com_cur;
                // Rotate the bind pose position relative to the CoM
                auto pos_bind_rot = R * pos_bind;
                // Our goal position
                auto goal = com_cur + pos_bind_rot;
                // Number of clusters this particle is a member of
                auto numClusters = NUMBER_OF_CLUSTERS(idx);
                auto correction = (goal - s.predicted_position[idx]) * stiffness;
                // The correction must be divided by the number of clusters this particle is a member of
                s.predicted_position[idx] += (1.0f / (float)numClusters) * correction;
                s.goal_position[idx] = goal;
            }

            s.predicted_orientation[i] = R;
        }

        END_BENCHMARK();
        PRINT_BENCHMARK_RESULT(_log);
    }

    void do_one_iteration_of_fixed_constraint_resolution(System_State& s, float phdt) override {
        for (auto i : s.fixed_particles) {
            s.predicted_position[i] = s.bind_pose[i];
        }
    }

    void do_one_iteration_of_distance_constraint_resolution(System_State& s, float phdt) override {
        auto const N = particle_count(s);
        for (unsigned i = 0; i < N; i++) {
            auto& neighbors = s.edges[i];
            auto w1 = 1 / mass_of_particle(s, i);
            for (auto j : neighbors) {
                auto w2 = 1 / mass_of_particle(s, j);
                auto w = w1 + w2;

                auto n = s.predicted_position[j] - s.predicted_position[i];
                auto d = glm::length(n);
                n = glm::normalize(n);
                auto restLength = glm::length(s.bind_pose[j] - s.bind_pose[i]);

                // TODO(danielm): get this value from params
                auto stiffness = 1.0f;
                auto corr = stiffness * n * (d - restLength) / w;

                s.predicted_position[i] += w1 * corr;
                s.predicted_position[j] += -w2 * corr;
            }
        }
    }

    Vector<Collision_Constraint> collision_constraints;

    void generate_collision_constraints(System_State& s) override {
        DECLARE_BENCHMARK_BLOCK();
        BEGIN_BENCHMARK();
        auto N = particle_count(s);

        collision_constraints.clear();

        for (auto& coll : s.colliders_sdf) {
            // Skip unused collider slots
            if (!coll.used) continue;

            for (auto i = 0ull; i < N; i++) {
                auto const start = s.position[i];
                auto thru = s.predicted_position[i];
                auto const dir = thru - start;

                auto coll_fun = [&](Vec3 const& sp) -> float {
                    coll.sp->set_value(sp);
                    return coll.expr->evaluate();
                };

                sdf::raymarch(coll_fun, 32, start, dir, 0.05f, 0.0f, 1.0f, [&](float dist) {
                    auto intersect = start + dist * dir;
                    auto normal = sdf::normal(coll_fun, intersect);
                    Collision_Constraint C;
                    C.intersect = intersect;
                    C.normal = normal;
                    C.pidx = i;
                    C.depth = length(intersect - thru);
                    collision_constraints.push_back(C);
                });
            }
        }

        for (auto const &coll : s.colliders_mesh) {
            if (!coll.used) continue;

            for (auto i = 0ull; i < N; i++) {
                auto const start = s.position[i];
                auto thru = s.predicted_position[i];
                auto const dir = thru - start;

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
                    if (!intersect_ray_triangle(xp, t, start, dir, v0, v1, v2) || t > 1) {
                        continue;
                    }
                    Collision_Constraint C;
                    C.intersect = Vec4(xp, 0);
                    C.depth = length(C.intersect - thru);
                    // C.intersect = v0;
                    auto n0 = coll.normals[ni0];
                    auto n1 = coll.normals[ni1];
                    auto n2 = coll.normals[ni2];
                    C.normal = Vec4(normalize(n0 + n1 + n2), 0);
                    C.pidx = i;

                    collision_constraints.push_back(C);

                    CHECK_VISUALIZER_PRESENT_DEBUG(
                        _visualizer->draw_intersection(start, thru, xp, v0, v1, v2, C.normal)
                    );
                }
            }
        }

        END_BENCHMARK();
        PRINT_BENCHMARK_RESULT_MASKED(_log, 0xF);
    }

    void do_one_iteration_of_collision_constraint_resolution(System_State& s, float phdt) override {
        DECLARE_BENCHMARK_BLOCK();
        BEGIN_BENCHMARK();
        for (auto& C : collision_constraints) {
            auto x = s.predicted_position[C.pidx] - C.intersect;
            auto n = C.normal;
            auto d = 0.5f;
            auto n_len = length(n);
            auto lambda = (dot(n, x) - d) / (n_len * n_len);
            auto corr = -lambda * n;
            s.predicted_position[C.pidx] += corr;
        }


        // Apply friction
        for (auto &C : collision_constraints) {
            // Relative tangential displacement:
            // (predicted position - previous position) projected onto the collider plane
            // To calculate this, first we project the vector onto the normal vector;
            // this new vector is then subtracted from the displacement vector
            auto dx = s.predicted_position[C.pidx] - s.position[C.pidx];
            auto dx_tan = dx - dot(dx, C.normal) * C.normal;

            // The correction vector is:
            //   if len(dx_tan) is less than the static friction coefficient: dx_tan itself
            //   otherwise: dx_tan * min((mu_k * d) / len(dx_tan), 1)
            // multiplied by w_i / (w_i + w_j).
            // Where:
            // - d is the penetration depth and
            // - mu_k, mu_s are the coefficients of kinetic and static friction,
            //   respectively.
            // - w_i is the inverse mass of the particle
            // - w_j is the inverse mass of the other thing
            //
            // In case of a collision with a static collider, we pretend as if the
            // collider body had infinite mass, which means that w_j will be 0,
            // making that constant have a value of 1.

            // Let's pretend that every object in the universe is made of wood
            float mu_s = 0.30f;
            float mu_k = 0.43f;

            auto len_dx_tan = length(dx_tan);

            glm::vec4 corr;
            if (len_dx_tan < mu_s * C.depth) {
                corr = dx_tan;
            } else {
                corr = dx_tan * glm::min(1.0f, (mu_k * C.depth) / len_dx_tan);
            }

            s.predicted_position[C.pidx] -= corr;
        }

        END_BENCHMARK();
        PRINT_BENCHMARK_RESULT_MASKED(_log, 0xF);
    }

protected:
    ILogger *_log;
    sb::IDebug_Visualizer *_visualizer;
};

sb::Unique_Ptr<ICompute_Backend> Make_Reference_Backend(ILogger* logger) {
    return std::make_unique<Compute_CPU_Single_Threaded>(logger);
}
