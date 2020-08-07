// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: iterators
//

#include "stdafx.h"
#include "softbody.h"
#include "s_simulation.h"
#include "m_utils.h"

/**
 * A base class for iterating over one-to-many maps like a Map<K, Vector<V>>.
 *
 * TODO(danielm): this is currently hard-coded for Map<unsigned, Vector<unsigned>>
 */
class One_To_Many_Relation_Iterator : public sb::Relation_Iterator {
public:
    using Map_Getter = std::function<Map<unsigned, Vector<unsigned>>&()>;
    using Relation_Factory = std::function<sb::Relation(unsigned lhs, unsigned rhs)>;
    One_To_Many_Relation_Iterator(Map_Getter mg, Relation_Factory rf)
        : get_map(mg), make_relation(rf) {
        p_iter = get_map().cbegin();
        p_end = get_map().cend();

        if (p_iter != p_end) {
            pv_iter = p_iter->second.cbegin();
            pv_end = p_iter->second.cend();

            while (p_iter != p_end && pv_iter == pv_end) {
                ++p_iter;

                if (p_iter != p_end) {
                    pv_iter = p_iter->second.cbegin();
                    pv_end = p_iter->second.cend();
                }
            }
        }
    }

private:
    void step() override {
        if (p_iter != p_end) {
            if (pv_iter != pv_end) {
                ++pv_iter;

                while (p_iter != p_end && pv_iter == pv_end) {
                    ++p_iter;

                    if (p_iter != p_end) {
                        pv_iter = p_iter->second.cbegin();
                        pv_end = p_iter->second.cend();
                    }
                }
            }
        }
    }

    bool ended() const override {
        return p_iter == p_end;
    }

    sb::Relation get() const override {
        assert(!ended());

        return make_relation(p_iter->first, *pv_iter);
    }

    using Particle_Iter = Map<unsigned, Vector<unsigned>>::const_iterator;
    using Particle_Vector_Iter = Vector<unsigned>::const_iterator;

    Map_Getter get_map;
    Relation_Factory make_relation;

    Particle_Iter p_iter, p_end;
    Particle_Vector_Iter pv_iter, pv_end;
};

/**
 * A base class for iterating over one-to-one maps like a Map<K, V>.
 *
 * TODO(danielm): this is currently hard-coded for Map<unsigned, unsigned>
 */
class One_To_One_Relation_Iterator : public sb::Relation_Iterator {
public:
    using Map_Getter = std::function<Map<unsigned, unsigned>& ()>;
    using Relation_Factory = std::function<sb::Relation(unsigned, unsigned)>;
    One_To_One_Relation_Iterator(Map_Getter mg, Relation_Factory rf) : make_relation(rf) {
        iter = mg().cbegin();
        end = mg().cend();
    }

private:
    virtual void step() override {
        if (iter != end) {
            ++iter;
        }
    }

    virtual bool ended() const override {
        return iter == end;
    }

    virtual sb::Relation get() const override {
        return make_relation(iter->first, iter->second);
    }

    using Iterator = Map<unsigned, unsigned>::const_iterator;
    Iterator iter, end;
    Relation_Factory make_relation;
};

/**
 * A base class for iterating over particles.
 */
class Particle_Iterator : public sb::Particle_Iterator {
public:
    using Particle_Count_Getter = std::function<size_t()>;
    using Particle_Factory = std::function<sb::Particle(size_t pidx)>;

    /**
     * @param pcg A callback through which this iterator will query the number
     * of particles in the system.
     * @param pf A function that provided with a particle index will return an
     * sb::Particle struct describing the particle.
     */
    Particle_Iterator(Particle_Count_Getter pcg, Particle_Factory pf)
        : idx(0), get_count(pcg), make_particle(pf) {
    }
private:
    void step() override {
        if (!ended()) {
            idx++;
        }
    }

    bool ended() const override {
        auto const N = get_count();
        return idx >= N;
    }

    sb::Particle get() const override {
        assert(!ended());

        return make_particle(idx);
    }

    size_t idx;
    Particle_Count_Getter get_count;
    Particle_Factory make_particle;
};

sb::Unique_Ptr<sb::Relation_Iterator> Softbody_Simulation::get_apical_relations() {
    auto get_map = [&]() -> decltype(s.apical_child)& { return s.apical_child; };
    auto make_relation = [&](unsigned lhs, unsigned rhs) {
        return sb::Relation {
            lhs,
            s.position[lhs],
            rhs,
            s.position[rhs],
        };
    };

    return std::make_unique<One_To_One_Relation_Iterator>(get_map, make_relation);
}

#define MAKE_PARTICLE_FACTORY(position_source) \
    [&](size_t pidx) {                                                          \
        sb::Particle ret;                                                       \
        ret.id = pidx;                                                          \
        ret.position = s.position_source[pidx];                                 \
        ret.orientation = s.orientation[pidx];                                  \
        ret.size = s.size[pidx];                                                \
        get_head_and_tail_of_particle(                                          \
            ret.position, longest_axis_normalized(ret.size), ret.orientation,   \
            &ret.start, &ret.end                                                \
        );                                                                      \
                                                                                \
        return ret;                                                             \
    };

sb::Unique_Ptr<sb::Particle_Iterator> Softbody_Simulation::get_particles() {
    auto pcg = [&]() { return s.position.size(); };
    auto pf = MAKE_PARTICLE_FACTORY(position);

    return std::make_unique<Particle_Iterator>(pcg, pf);
}

sb::Unique_Ptr<sb::Particle_Iterator> Softbody_Simulation::get_particles_with_goal_positions() {
    auto pcg = [&]() { return s.position.size(); };
    auto pf = MAKE_PARTICLE_FACTORY(goal_position);

    return std::make_unique<Particle_Iterator>(pcg, pf);
}

sb::Unique_Ptr<sb::Particle_Iterator> Softbody_Simulation::get_particles_with_predicted_positions() {
    auto pcg = [&]() { return s.position.size(); };
    auto pf = MAKE_PARTICLE_FACTORY(predicted_position);

    return std::make_unique<Particle_Iterator>(pcg, pf);
}

sb::Unique_Ptr<sb::Particle_Iterator> Softbody_Simulation::get_centers_of_masses() {
    auto pcg = [&]() { return s.position.size(); };
    auto pf = MAKE_PARTICLE_FACTORY(center_of_mass);

    return std::make_unique<Particle_Iterator>(pcg, pf);
}

sb::Unique_Ptr<sb::Relation_Iterator> Softbody_Simulation::get_lateral_relations() {
    auto get_map = [&]() -> decltype(s.lateral_bud)& { return s.lateral_bud; };
    auto make_relation = [&](unsigned lhs, unsigned rhs) {
        return sb::Relation {
            lhs,
            s.position[lhs],
            rhs,
            s.position[rhs],
        };
    };

    return std::make_unique<One_To_One_Relation_Iterator>(get_map, make_relation);
}

sb::Unique_Ptr<sb::Relation_Iterator> Softbody_Simulation::get_connections() {
    auto get_map = [&]() -> decltype(s.edges)& { return s.edges; };
    auto make_relation = [&](unsigned lhs, unsigned rhs) {
        return sb::Relation {
            lhs,
            s.position[lhs],
            rhs,
            s.position[rhs],
        };
    };

    return std::make_unique<One_To_Many_Relation_Iterator>(get_map, make_relation);
}

sb::Unique_Ptr<sb::Relation_Iterator> Softbody_Simulation::get_predicted_connections() {
    auto get_map = [&]() -> decltype(s.edges)& { return s.edges; };
    auto make_relation = [&](unsigned lhs, unsigned rhs) {
        return sb::Relation {
            lhs,
            s.predicted_position[lhs],
            rhs,
            s.predicted_position[rhs],
        };
    };

    return std::make_unique<One_To_Many_Relation_Iterator>(get_map, make_relation);
}

sb::Unique_Ptr<sb::Particle_Iterator> Softbody_Simulation::get_particles_in_bind_pose() {
    auto pcg = [&]() { return s.position.size(); };
    auto pf = MAKE_PARTICLE_FACTORY(bind_pose);

    return std::make_unique<Particle_Iterator>(pcg, pf);
}

sb::Unique_Ptr<sb::Relation_Iterator> Softbody_Simulation::get_connections_in_bind_pose() {
    auto get_map = [&]() -> decltype(s.edges)& { return s.edges; };
    auto make_relation = [&](unsigned lhs, unsigned rhs) {
        return sb::Relation {
            lhs,
            s.bind_pose[lhs],
            rhs,
            s.bind_pose[rhs],
        };
    };

    return std::make_unique<One_To_Many_Relation_Iterator>(get_map, make_relation);
}
