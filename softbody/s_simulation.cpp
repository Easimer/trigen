// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: softbody simulation
//

#include "stdafx.h"
#include "softbody.h"
#include <cstdlib>

// nocheckin 
#include <imgui.h>

#define PHYSICS_STEP (1.0f / 25.0f)
#define SIM_SIZE_LIMIT (4)

#define DISABLE_GROWTH
#define DISABLE_PHOTOTROPISM

using Vec3 = glm::vec3;
using Mat3 = glm::mat3;
using Quat = glm::quat;

template<typename T>
using Vector = std::vector<T>;

template<typename K, typename V>
using Map = std::unordered_map<K, V>;

struct Particle_Group {
    unsigned owner;
    float owner_mass;
    Vector<unsigned> neighbors;
    Vector<float> masses;
    float W;
    Vec3 c, c_rest;
    Mat3 orient;
};

static float randf() {
    return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

static Vec3 longest_axis_normalized(Vec3 const& v) {
    auto idx =
        (v.x > v.y) ?
        ((v.x > v.z) ? (0) : (2))
        :
        ((v.y > v.z ? (1) : (2)));
    Vec3 ret(0, 0, 0);
    ret[idx] = 1;
    return glm::normalize(ret);
}

static void get_head_and_tail_of_particle(
    Vec3 const& pos,
    Vec3 const& longest_axis,
    Quat const& orientation,
    Vec3* out_head,
    Vec3* out_tail
) {
    auto const axis_rotated = orientation * longest_axis * glm::inverse(orientation);
    auto const axis_rotated_half = 0.5f * axis_rotated;
    *out_head = pos - axis_rotated_half;
    *out_tail = pos + axis_rotated_half;
}

struct Softbody_Simulation {
    Vector<Vec3> position;
    Vector<Vec3> velocity;
    Vector<Vec3> angular_velocity;
    Vector<Vec3> rest_position;
    Vector<Vec3> goal_position;
    Vector<Vec3> center_of_mass;
    Vector<Vec3> rest_center_of_mass;
    Vector<Vec3> size;
    Vector<Quat> orientation;
    Vector<float> density;
    Map<unsigned, Vector<unsigned>> edges;
    Map<unsigned, unsigned> apical_child;
    Map<unsigned, unsigned> lateral_bud;

    Vector<Vec3> predicted_position;

    float time_accumulator = 0.0f;

    Vec3 light_source = Vec3(0, 0, 0);

    unsigned add_particle(Vec3 const& p_pos, Vec3 const& p_size, float p_density) {
        assert(p_density >= 0.0f && p_density <= 1.0f);
        Vec3 zero(0, 0, 0);
        unsigned const index = position.size();
        position.push_back(p_pos);
        rest_position.push_back(p_pos);
        velocity.push_back(zero);
        angular_velocity.push_back(zero);
        goal_position.push_back(p_pos);
        center_of_mass.push_back(zero);
        rest_center_of_mass.push_back(zero);
        size.push_back(p_size);
        density.push_back(p_density);
        orientation.push_back(Quat(1.0f, 0.0f, 0.0f, 0.0f));
        edges[index] = {};

        return index;
    }

    void connect_particles(unsigned a, unsigned b) {
        assert(a < position.size());
        assert(b < position.size());

        edges[a].push_back(b);
        edges[b].push_back(a);
    }

    float mass_of_particle(unsigned i) {
        auto const d_i = density[i];
        auto const s_i = size[i];
        auto const m_i = (4.f / 3.f) * glm::pi<float>() * s_i.x * s_i.y * s_i.z * d_i;
        return m_i;
    }

    void predict_positions(float dt) {
        ImGui::Begin("Predict_Positions", NULL, ImGuiWindowFlags_NoCollapse);

        predicted_position.clear();
        for (auto i = 0ull; i < position.size(); i++) {
            ImGui::Text("particle %ull", i);

            auto a = Vec3();
            auto const a_g = Vec3(0, -1, 0);
            auto const goal_dir = (goal_position[i] - position[i]);
            auto const a_goal = goal_dir * dt;
            a += a_g;
            a += 16.0f * a_goal;
            auto const v = velocity[i] + dt * a;
            velocity[i] = v;
            // Vec3 const x_p = system.position[i] + dt * system.velocity[i] + (dt * dt / 2) * a + 2.0f * goal_dir * dt;
            Vec3 const x_p = position[i] + dt * velocity[i] + (dt * dt / 2) * a;
            predicted_position.push_back(x_p);

#ifndef DISABLE_PHOTOTROPISM
            // Phototropism
            auto v_forward = glm::normalize(orientation[i] * Vec3(0, 1, 0) * glm::inverse(orientation[i]));
            auto v_light = glm::normalize(light_source - position[i]);
            auto v_light_axis = glm::cross(v_light, v_forward);
            ImGui::InputFloat3("v_forward", &v_forward[0]);
            ImGui::InputFloat3("v_light", &v_light[0]);
            ImGui::InputFloat3("v_light_axis", &v_light_axis[0]);
            auto O = 0.0f; // TODO: detect occlusion, probably by shadow mapping
            auto eta = 0.5f; // suspectability to phototropism
            auto angle_light = (1 - O) * eta * dt;
            auto Q_light = glm::normalize(Quat(angle_light, v_light_axis));
            // angular_velocity[i] = Q_light * angular_velocity[i] * glm::inverse(Q_light);
            orientation[i] = Q_light * orientation[i];
#endif /* !defined(DISABLE_PHOTOTROPISM) */

            auto ang_vel = angular_velocity[i];
            auto orient = orientation[i];
            auto len_ang_vel = glm::length(ang_vel);
            Quat orient_temp = orient;
            if (len_ang_vel >= glm::epsilon<float>()) {
                auto comp = (len_ang_vel * dt) / 2;
                 orient_temp = Quat(glm::cos(comp), (ang_vel / len_ang_vel) * glm::sin(comp));
            }

            auto delta_orient_angle = glm::angle(orient_temp * glm::inverse(orient));
            auto delta_orient_axis = glm::axis(orient_temp * glm::inverse(orient));

            if (glm::abs(delta_orient_angle) >= glm::epsilon<float>()) {
                angular_velocity[i] = (delta_orient_angle / dt) * delta_orient_axis;
            }

            orientation[i] = orient_temp;


            ImGui::InputFloat4("orientation", &orientation[i][0]);
        }

        ImGui::End();
    }

    Mat3 polar_decompose_r(Mat3 const& A) {
        glm::mat4 A4(A);
        glm::vec3 scale;
        glm::quat rotate;
        glm::vec3 translate;
        glm::vec3 skew;
        glm::vec4 perspective;
        auto v2 = A4[2];

        if (v2.z == 0.0f) {
            // HACKHACKHACK: mivel meg csak sikban tesztelunk, ezert a matrix Z oszlopa
            // nullvektor lesz. Ez azt okozza, hogy a matrix szingularis lesz es nem tudja
            // a glm::decompose dekompozicionalni (hiaba lenne neki).
            // Ha jol hiszem ez a problema meg fog oldodni, amint tetraederekbol epitjuk fel
            // a modellt.
            // Nem tudom mi lesz akkor, amikor reszecskelanc lesz a modell.
            v2.z = 1;
            A4[2] = v2;
        }

        A4[3] = glm::vec4(0, 0, 0, 1);
        if (glm::decompose(A4, scale, rotate, translate, skew, perspective)) {
            rotate = glm::conjugate(rotate);
            return (Mat3)rotate;
        } else {
            assert(0);
        }
    }


    void calculate_orientation_matrix(Particle_Group* group) {
        /*
        auto const A = [this, group]() {
            Mat3 ret = group->owner_mass *
                glm::outerProduct(
                                  (position[group->owner] - group->c),
                                  (rest_position[group->owner] - group->c_rest));

            for(unsigned i = 0; i < group->neighbors.size(); i++) {
                auto const pidx = group->neighbors[i];
                auto const pos = position[pidx];
                auto const pos_rest = rest_position[pidx];
                auto const m_i = group->masses[i];
                auto A_i = m_i *
                    glm::outerProduct(
                                      (pos - group->c),
                                      (pos_rest - group->c_rest));
                ret += A_i;
            }

            return ret;
        }();
        */

        auto calcA_pq = [this, group](unsigned pidx) -> Mat3 {
            auto const p_i = position[pidx] - center_of_mass[pidx];
            auto const q_i = rest_position[pidx] - rest_center_of_mass[pidx];
            auto const m_i = mass_of_particle(pidx);

            return m_i * glm::outerProduct(p_i, q_i);
        };

        auto is_null_matrix = [](Mat3 const& m) -> bool {
            float sum = 0;
            for (int i = 0; i < 3; i++) {
                sum += m[i].length();
            }

            return sum < glm::epsilon<float>();
        };

        auto const A = [this, group, calcA_pq, is_null_matrix]() -> Mat3 {
            auto sum = calcA_pq(group->owner);
            if (is_null_matrix(sum)) {

                for (auto i : group->neighbors) {
                    sum += calcA_pq(i);
                }

                return sum;
            } else {
                // Amennyiben egy reszecske pontosan a csoportjanak tomegkozeppontjaban van,
                // akkor az orientacios matrixa a nullmatrix lesz.
                // Mivel az olyasmit a polar_decompose_r nem szereti, es jelentestartalmilag a nullmatrix
                // es az identitasmatrix ugyanaz, ezert az identitasmatrixot adjuk vissza
                return Mat3(1.0f);
            }

            return sum;
        }();

        auto const& owner_size = size[group->owner];

        auto const R = polar_decompose_r(A);

        group->orient = R;
    }


    void simulate_group(unsigned pidx, float dt) {
        auto const& neighbors = edges[pidx];
        auto& owner_pos = position[pidx];
        auto& owner_rest_pos = rest_position[pidx];

        auto const [owner_mass, masses] = [&]() {
            Vector<float> masses;
            for (auto i : neighbors) {
                masses.push_back(mass_of_particle(i));
            }
            auto owner_mass = mass_of_particle(pidx);
            return std::make_tuple(owner_mass, masses);
        }();

        auto const W = [owner_mass = owner_mass, masses = masses]() {
            float sum = 0;
            for (auto m : masses) {
                sum += m;
            }
            sum += owner_mass;
            return sum;
        }();

        auto const [c, c_rest] = [&]() {
            auto c = Vec3();
            auto c_rest = Vec3();

            for (auto i : neighbors) {
                auto const m_i = mass_of_particle(i);
                c += (m_i / W) * position[i];
                c_rest += (m_i / W) * rest_position[i];
            }
            auto const m_c = mass_of_particle(pidx);
            // c = (m_c / W) * position[pidx];
            // c_rest = (m_c / W) * rest_position[pidx];
            c += (m_c / W) * owner_pos;
            c_rest += (m_c / W) * owner_rest_pos;

            return std::make_tuple(c, c_rest);
        }();

        auto group = Particle_Group{ pidx, owner_mass, neighbors, masses, W, c, c_rest, Mat3() };
        calculate_orientation_matrix(&group);
        auto const x_t = group.orient * (owner_rest_pos - group.c_rest) + group.c;
        auto const& owner_predicted = predicted_position[pidx];
        // auto const v = (owner_predicted - owner_pos) / dt;
        // velocity[pidx] = v;
        owner_pos = owner_predicted;

        center_of_mass[pidx] = c;
        rest_center_of_mass[pidx] = c_rest;

        goal_position[pidx] = x_t;

#ifndef DISABLE_GROWTH
        // TODO(danielm): novekedesi rata
        // Novesztjuk az agat
        auto g = 4.0f; // 1/sec
        auto prob_branching = 0.25f;
        auto& r = size[pidx].x;
        if (r < 1.5f) {
            size[pidx] += Vec3(g * dt, g * dt, 0);

            // Ha tulleptuk a meret-limitet, novesszunk uj agat
            if (r >= 1.5f && position.size() < SIM_SIZE_LIMIT) {
                auto lateral_chance = randf();
                constexpr auto new_size = Vec3(0.5f, 0.5f, 2.0f);
                auto longest_axis = longest_axis_normalized(size[pidx]);
                auto new_longest_axis = longest_axis_normalized(new_size);
                if (lateral_chance < prob_branching) {
                    // Oldalagat novesszuk
                    auto bud_rot_offset = glm::angleAxis(-45.0f, Vec3(0, 0, 1));
                    auto lateral_orientation = bud_rot_offset * orientation[pidx];
                    auto l_pos = position[pidx]
                        + orientation[pidx] * (longest_axis / 2.0f) * glm::inverse(orientation[pidx])
                        + lateral_orientation * (new_longest_axis / 2.0f) * glm::inverse(lateral_orientation);
                    auto l_idx = add_particle(l_pos, new_size, 1.0f);
                    lateral_bud[pidx] = l_idx;
                    connect_particles(pidx, l_idx);
                    orientation[l_idx] = lateral_orientation;
                }

                // Csucsot novesszuk
                auto pos = position[pidx]
                    + orientation[pidx] * (longest_axis /2.0f) * glm::inverse(orientation[pidx])
                    + orientation[pidx] * (new_longest_axis /2.0f) * glm::inverse(orientation[pidx]);
                auto a_idx = add_particle(pos, new_size, 1.0f);
                apical_child[pidx] = a_idx;
                    connect_particles(pidx, a_idx);
            }
        }
#endif /* !defined(DISABLE_GROWTH) */
    }
};

struct Particle_Iterator_Impl : public sb::Particle_Iterator {
    Softbody_Simulation* sim;
    unsigned idx;

    Particle_Iterator_Impl(Softbody_Simulation* sim) : sim(sim), idx(0) {}

    virtual void release() override {
        delete this;
    }

    virtual void step() override {
        auto const N = sim->position.size();
        auto end_state = (idx >= sim->position.size());
        if (!end_state) {
            idx++;
        }
    }

    virtual bool ended() const override {
        return idx >= sim->position.size();
    }

    virtual sb::Particle get() const override {
        assert(idx < sim->position.size());
        Vec3 head, tail;
        Vec3 const pos = sim->position[idx];
        Vec3 const size = sim->size[idx];
        auto const orientation = sim->orientation[idx];
        get_head_and_tail_of_particle(pos, longest_axis_normalized(size), orientation, &head, &tail);

        return sb::Particle {
            idx,
            pos,
            head, tail,
        };
    }
};


Softbody_Simulation* sb::create_simulation(Config const& configuration) {
    auto ret = new Softbody_Simulation;

    auto idx_root = ret->add_particle(configuration.seed_position, Vec3(1, 1, 2), 1);
    auto idx_up = ret->add_particle(configuration.seed_position + Vec3(0, 0, 2), Vec3(1, 1, 2), 1);
    auto idx_down = ret->add_particle(configuration.seed_position - Vec3(0, 0, 2), Vec3(1, 1, 2), 1);
    ret->connect_particles(idx_root, idx_up);
    ret->connect_particles(idx_root, idx_down);

    return ret;
}

void sb::destroy_simulation(Softbody_Simulation* s) {
    assert(s != NULL);

    if (s != NULL) {
        delete s;
    }
}

void sb::set_light_source_position(Softbody_Simulation* s, Vec3 const& pos) {
    assert(s != NULL);
    if (s != NULL) {
        s->light_source = pos;
    }
}

void sb::step(Softbody_Simulation* s, float delta_time) {
    assert(s != NULL);
    if (s != NULL) {
        s->predict_positions(delta_time);

        s->time_accumulator += delta_time;

        // Nem per-frame szimulalunk, hanem fix idokozonkent, hogy ne valjon
        // instabilla a szimulacio
        if (s->time_accumulator > PHYSICS_STEP) {
            auto phdt = s->time_accumulator;
            auto p0 = s->position[0];
            for (auto idx = 0ull; idx < s->predicted_position.size(); idx++) {
                s->simulate_group(idx, phdt);
            }
            s->position[0] = p0;

            s->time_accumulator = 0;
        }
    }
}

sb::Particle_Iterator* sb::get_particles(Softbody_Simulation* s) {
    assert(s != NULL);
    if (s != NULL) {
        return new Particle_Iterator_Impl(s);
    } else {
        return NULL;
    }
}

class Apical_Relation_Iterator : public sb::Relation_Iterator {
public:
    Apical_Relation_Iterator(Softbody_Simulation* s) : s(s) {
        iter = s->apical_child.begin();
        end = s->apical_child.end();
    }
private:
    Softbody_Simulation* s;
    typename decltype(s->apical_child)::const_iterator iter;
    typename decltype(s->apical_child)::const_iterator end;

    virtual void release() override {
        delete this;
    }

    virtual void step() override {
        if (iter != end) {
            iter++;
        }
    }

    virtual bool ended() const override {
        return iter == end;
    }

    virtual sb::Relation get() const override {
        assert(!ended());
        return sb::Relation {
            iter->first,
            s->position[iter->first],
            iter->second,
            s->position[iter->second],
        };
    }
};

class Lateral_Relation_Iterator : public sb::Relation_Iterator {
public:
    Lateral_Relation_Iterator(Softbody_Simulation* s) : s(s) {
        iter = s->lateral_bud.begin();
        end = s->lateral_bud.end();
    }
private:
    Softbody_Simulation* s;
    typename decltype(s->lateral_bud)::const_iterator iter;
    typename decltype(s->lateral_bud)::const_iterator end;

    virtual void release() override {
        delete this;
    }

    virtual void step() override {
        if (iter != end) {
            iter++;
        }
    }

    virtual bool ended() const override {
        return iter == end;
    }

    virtual sb::Relation get() const override {
        assert(!ended());
        return sb::Relation {
            iter->first,
            s->position[iter->first],
            iter->second,
            s->position[iter->second],
        };
    }
};

sb::Relation_Iterator* sb::get_apical_relations(Softbody_Simulation* s) {
    assert(s != NULL);
    return new Apical_Relation_Iterator(s);
}

sb::Relation_Iterator* sb::get_lateral_relations(Softbody_Simulation* s) {
    assert(s != NULL);
    return new Lateral_Relation_Iterator(s);
}
