// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: iterators
//

#include "stdafx.h"
#include "softbody.h"
#include "s_simulation.h"
#include "m_utils.h"

namespace sb {
    template<typename Getter>
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
            Getter G{ sim };
            Vec3 const pos = G.position(idx);
            Vec3 const size = G.size(idx);
            auto const orientation = G.orientation(idx);
            G.head_and_tail(pos, longest_axis_normalized(size), orientation, &head, &tail);
            // get_head_and_tail_of_particle(pos, longest_axis_normalized(size), orientation, &head, &tail);

            return sb::Particle{
                idx,
                pos,
                orientation,
                size,
                head, tail,
            };
        }
    };

    // Used as the template argument of Particle_Iterator_Impl
    struct Normal_Particle_Getter {
        Softbody_Simulation* s;

        Vec3 position(unsigned idx) {
            return s->position[idx];
        }

        Vec3 size(unsigned idx) {
            return s->size[idx];
        }

        Quat orientation(unsigned idx) {
            return s->orientation[idx];
        }

        void head_and_tail(Vec3 const& pos, Vec3 const& axis, Quat const& orientation, Vec3* head, Vec3* tail) {
            get_head_and_tail_of_particle(pos, axis, orientation, head, tail);
        }
    };

    // Used as the template argument of Particle_Iterator_Impl
    // Instead of the actual position, this will make the iterator return the
    // goal position of the particle.
    struct Goal_Position_Particle_Getter : public Normal_Particle_Getter {
        Vec3 position(unsigned idx) {
            return s->goal_position[idx];
        }
    };

    struct Center_Of_Mass_Getter : public Normal_Particle_Getter {
        Vec3 position(unsigned idx) {
            return s->center_of_mass[idx];
        }
    };

    Particle_Iterator* get_particles(Softbody_Simulation* s) {
        assert(s != NULL);
        if (s != NULL) {
            return new Particle_Iterator_Impl<Normal_Particle_Getter>(s);
        } else {
            return NULL;
        }
    }

    Particle_Iterator* get_particles_with_goal_position(Softbody_Simulation* s) {
        assert(s != NULL);
        if (s != NULL) {
            return new Particle_Iterator_Impl<Goal_Position_Particle_Getter>(s);
        } else {
            return NULL;
        }
    }

    Particle_Iterator* get_centers_of_masses(Softbody_Simulation* s) {
        assert(s != NULL);
        if (s != NULL) {
            return new Particle_Iterator_Impl<Center_Of_Mass_Getter>(s);
        } else {
            return NULL;
        }
    }

    class Apical_Relation_Iterator : public Relation_Iterator {
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

        virtual Relation get() const override {
            assert(!ended());
            return Relation{
                iter->first,
                s->position[iter->first],
                iter->second,
                s->position[iter->second],
            };
        }
    };

    class Lateral_Relation_Iterator : public Relation_Iterator {
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

        virtual Relation get() const override {
            assert(!ended());
            return Relation{
                iter->first,
                s->position[iter->first],
                iter->second,
                s->position[iter->second],
            };
        }
    };

    Relation_Iterator* get_apical_relations(Softbody_Simulation* s) {
        assert(s != NULL);
        return new Apical_Relation_Iterator(s);
    }

    Relation_Iterator* get_lateral_relations(Softbody_Simulation* s) {
        assert(s != NULL);
        return new Lateral_Relation_Iterator(s);
    }

    Relation_Iterator* get_connections(Softbody_Simulation* s) {
        assert(s != NULL);

        class Connection_Iterator : public Iterator<Relation> {
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

                    if (iter == end) {
                        pidx++;
                    }
                }
            }

            virtual bool ended() const override {
                return pidx == s->position.size();
            }

            virtual Relation get() const override {
                assert(!ended());
                return Relation{
                    pidx,
                    s->position[pidx],
                    *iter,
                    s->position[*iter],
                };
            }
        };

        return new Connection_Iterator(s);
    }
}
