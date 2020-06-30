// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: softbody simulation
//

#include "stdafx.h"
#include "softbody.h"
#include <cstdlib>

#define PHYSICS_STEP (1.0f / 25.0f)
#define SIM_SIZE_LIMIT (128)

// #define DISABLE_GROWTH
// #define DISABLE_PHOTOTROPISM

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

template<typename T, typename Iterator>
static T sum(Iterator begin, Iterator end) {
    T ret = 0;

    while (begin != end) {
        ret += *begin++;
    }

    return ret;
}

class Deferred_Particle_Creator {

};

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

    bool assert_parallel = false;

    float time_accumulator = 0.0f;

    Vec3 light_source = Vec3(0, 0, 0);

    // Stores functions whose execution has been deferred until after the parallelized
    // part
    Mutex deferred_lock;
    Vector<Fun<void()>> deferred;

    unsigned add_particle(Vec3 const& p_pos, Vec3 const& p_size, float p_density) {
        assert(!assert_parallel);
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
        assert(!assert_parallel);
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
        predicted_position.clear();
        auto a_wind = Vec3(randf() * 0.05, 0, 0);
        for (auto i = 0ull; i < position.size(); i++) {

            auto a = Vec3();
            auto const a_g = Vec3(0, -1, 0);
            auto const goal_dir = (goal_position[i] - position[i]);
            auto const a_goal = goal_dir * dt;
            a += a_g;
            a += 64.0f * a_goal;
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
            auto O = 0.0f; // TODO: detect occlusion, probably by shadow mapping
            auto eta = 0.5f; // suspectability to phototropism
            auto angle_light = (1 - O) * eta * dt;
            auto Q_light = glm::normalize(Quat(angle_light, v_light_axis));
            angular_velocity[i] = Q_light * angular_velocity[i] * glm::inverse(Q_light);
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

            if (glm::abs(delta_orient_angle) >= 0.05f) {
                angular_velocity[i] = (delta_orient_angle / dt) * delta_orient_axis;
            }

            orientation[i] = orient_temp;
        }
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
        auto calculate_particle_matrix = [this](unsigned pidx) -> Mat3 {
            auto mass_i = this->mass_of_particle(pidx);
            auto const size = this->size[pidx];
            auto const a = size[0];
            auto const b = size[1];
            auto const c = size[2];
            auto const A_i = Mat3(a * a, 0, 0, 0, b * b, 0, 0, 0, c * c);
            return mass_i * (1/5.0f) * A_i;
        };

        auto calculate_group_matrix = [this, group, calculate_particle_matrix]() -> Mat3 {
            // sum (particle matrix + particle mass * (particle pos * particle rest pos)) - total group mass * (center * rest center)

            float const sum_masses = group->owner_mass + sum<float>(group->masses.begin(), group->masses.end());

            auto moment_sum =
                calculate_particle_matrix(group->owner)
                + this->mass_of_particle(group->owner) * glm::outerProduct(predicted_position[group->owner], rest_position[group->owner]);

            for (auto neighbor : group->neighbors) {
                moment_sum +=
                    calculate_particle_matrix(neighbor)
                    + this->mass_of_particle(neighbor) * glm::outerProduct(predicted_position[neighbor], rest_position[neighbor]);
            }

            return moment_sum - sum_masses * glm::outerProduct(group->c, group->c_rest);
        };

        auto is_null_matrix = [](Mat3 const& m) -> bool {
            float sum = 0;
            for (int i = 0; i < 3; i++) {
                sum += m[i].length();
            }

            return sum < glm::epsilon<float>();
        };

        auto const& owner_size = size[group->owner];

        auto const R = polar_decompose_r(calculate_group_matrix());

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
                c += (m_i / W) * predicted_position[i];
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
        auto g = 1.0f; // 1/sec
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
                    auto angle = randf() * glm::two_pi<float>();
                    auto x = 2 * randf() - 1;
                    auto y = 2 * randf() - 1;
                    auto z = 2 * randf() - 1;
                    auto axis = glm::normalize(Vec3(x, y, z));
                    auto bud_rot_offset = glm::angleAxis(angle, axis);
                    auto lateral_orientation = bud_rot_offset * orientation[pidx];
                    auto l_pos = position[pidx]
                        + orientation[pidx] * (longest_axis / 2.0f) * glm::inverse(orientation[pidx])
                        + lateral_orientation * (new_longest_axis / 2.0f) * glm::inverse(lateral_orientation);

                    auto func_add = [&, l_pos, new_size, pidx, lateral_orientation]() {
                        auto l_idx = add_particle(l_pos, new_size, 1.0f);
                        lateral_bud[pidx] = l_idx;
                        connect_particles(pidx, l_idx);
                        orientation[l_idx] = lateral_orientation;
                    };
                    Lock_Guard g(deferred_lock);
                    deferred.push_back(std::move(func_add));
                }

                // Csucsot novesszuk
                auto pos = position[pidx]
                    + orientation[pidx] * (longest_axis /2.0f) * glm::inverse(orientation[pidx])
                    + orientation[pidx] * (new_longest_axis /2.0f) * glm::inverse(orientation[pidx]);
                auto func_add = [&, pos, new_size, pidx]() {
                    auto a_idx = add_particle(pos, new_size, 1.0f);
                    apical_child[pidx] = a_idx;
                        connect_particles(pidx, a_idx);
                };
                Lock_Guard g(deferred_lock);
                deferred.push_back(std::move(func_add));
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

    auto o = configuration.seed_position;
    auto siz = Vec3(1, 1, 2);
    auto idx_root = ret->add_particle(o, siz, 1);
#ifdef DEBUG_TETRAHEDRON
    auto idx_t0 = ret->add_particle(o + Vec3(-1, 2, 1), siz, 1);
    auto idx_t1 = ret->add_particle(o + Vec3(+1, 2, 1), siz, 1);
    auto idx_t2 = ret->add_particle(o + Vec3( 0, 2, -1), siz, 1);

    ret->connect_particles(idx_root, idx_t0);
    ret->connect_particles(idx_root, idx_t1);
    ret->connect_particles(idx_root, idx_t2);
    ret->connect_particles(idx_t0, idx_t1);
    ret->connect_particles(idx_t1, idx_t2);
    ret->connect_particles(idx_t0, idx_t2);
#endif /* DEBUG_TETRAHEDRON */

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

static void stop_particle(Softbody_Simulation* s, unsigned pidx) {
    s->velocity[pidx] = Vec3(0, 0, 0);
    s->angular_velocity[pidx] = Vec3(0, 0, 0);
}

static void distributed_step(Softbody_Simulation* s, float dt, unsigned from, unsigned to) {
    for (auto idx = from; idx < to; idx++) {
        s->simulate_group(idx, dt);
    }
}

class range {
public:
    class iterator {
        friend class range;
    public:
        long int operator *() const { return i_; }
        const iterator& operator ++() { ++i_; return *this; }
        iterator operator ++(int) { iterator copy(*this); ++i_; return copy; }

        bool operator ==(const iterator& other) const { return i_ == other.i_; }
        bool operator !=(const iterator& other) const { return i_ != other.i_; }

        iterator() : i_(0) {}
    protected:
        iterator(long int start) : i_(start) {}

    private:
        unsigned long i_;
    };

    iterator begin() const { return begin_; }
    iterator end() const { return end_; }
    range(long int  begin, long int end) : begin_(begin), end_(end) {}
private:
    iterator begin_;
    iterator end_;
};

template<>
struct std::iterator_traits<range::iterator> {
    using difference_type = long int;
    using value_type = long int;
    using pointer = range::iterator*;
    using reference = range::iterator&;
    using iterator_category = std::forward_iterator_tag;
};

void sb::step(Softbody_Simulation* s, float delta_time) {
    assert(s != NULL);
    if (s != NULL) {
        //s->predict_positions(delta_time);

        s->time_accumulator += delta_time;

        // Nem per-frame szimulalunk, hanem fix idokozonkent, hogy ne valjon
        // instabilla a szimulacio
        while(s->time_accumulator > PHYSICS_STEP) {
            auto phdt = PHYSICS_STEP;
            auto p0 = s->position[0];

            s->predict_positions(phdt);

            if (s->time_accumulator > 8 * PHYSICS_STEP) {
                fprintf(stderr, "EXTREME LAG, ACC = %f\n", s->time_accumulator);
            }

            auto R = range(0, s->predicted_position.size());
            s->assert_parallel = true;
            std::for_each(std::execution::par, R.begin(), R.end(), [&](unsigned i) {
                s->simulate_group(i, phdt);
            });
            s->assert_parallel = false;

            for (auto& def_func : s->deferred) {
                def_func();
            }
            s->deferred.clear();
            /*
            for (auto idx = 0ull; idx < s->predicted_position.size(); idx++) {
                s->simulate_group(idx, phdt);
            }
            */
            s->position[0] = p0;

            s->time_accumulator -= PHYSICS_STEP;
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

sb::Relation_Iterator* sb::get_connections(Softbody_Simulation* s) {
    assert(s != NULL);

    class Connection_Iterator : public sb::Iterator<Relation> {
    public:
        Connection_Iterator(Softbody_Simulation* s) : s(s), pidx(0) {
            iter = s->edges[0].begin();
            end = s->edges[0].end();

            if (iter == end) {
                pidx++;
            }
        }
    private:
        Softbody_Simulation* s;
        unsigned pidx;
        typename Vector<unsigned>::const_iterator iter;
        typename Vector<unsigned>::const_iterator end;

        virtual void release() override {
            delete this;
        }

        virtual void step() override {
            if (pidx != s->position.size()) {
                if (iter != end) {
                    iter++;

                    if (iter == end) {
                        next_key();
                    }
                } else {
                    next_key();
                }
            }
        }

        void next_key() {
            pidx++;
            if (pidx != s->position.size()) {
                iter = s->edges[pidx].begin();
                end = s->edges[pidx].end();
            }
        }

        virtual bool ended() const override {
            return pidx == s->position.size();
        }

        virtual sb::Relation get() const override {
            assert(!ended());
            return sb::Relation{
                pidx,
                s->position[pidx],
                *iter,
                s->position[*iter],
            };
        }
    };

    return new Connection_Iterator(s);
}
