// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: iterators
//

#pragma once

#include "softbody.h"

/**
 * A base class for iterating over one-to-many maps like a Map<K, Vector<V>>.
 *
 * TODO(danielm): this is currently hard-coded for Map<index_t, Vector<index_t>>
 */
class One_To_Many_Relation_Iterator : public sb::Relation_Iterator {
public:
    using Map_Getter = std::function<Map<index_t, Vector<index_t>>&()>;
    using Relation_Factory = std::function<sb::Relation(index_t lhs, index_t rhs)>;
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

    using Particle_Iter = Map<index_t, Vector<index_t>>::const_iterator;
    using Particle_Vector_Iter = Vector<index_t>::const_iterator;

    Map_Getter get_map;
    Relation_Factory make_relation;

    Particle_Iter p_iter, p_end;
    Particle_Vector_Iter pv_iter, pv_end;
};

/**
 * A base class for iterating over one-to-one maps like a Map<K, V>.
 *
 * TODO(danielm): this is currently hard-coded for Map<index_t, index_t>
 */
class One_To_One_Relation_Iterator : public sb::Relation_Iterator {
public:
    using Map_Getter = std::function<Map<index_t, index_t>& ()>;
    using Relation_Factory = std::function<sb::Relation(index_t, index_t)>;
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

    using Iterator = Map<index_t, index_t>::const_iterator;
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

#define MAKE_PARTICLE_FACTORY(position_source) \
    [&](index_t pidx) {                                                         \
        sb::Particle ret;                                                       \
        ret.id = (unsigned)pidx;                                                \
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
