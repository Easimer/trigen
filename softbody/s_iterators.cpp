// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: iterators
//

#include "stdafx.h"
#include "softbody.h"
#include "s_simulation.h"
#include "m_utils.h"

namespace sb {
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

            return sb::Particle{
                idx,
                pos,
                head, tail,
            };
        }
    };

    Particle_Iterator* get_particles(Softbody_Simulation* s) {
        assert(s != NULL);
        if (s != NULL) {
            return new Particle_Iterator_Impl(s);
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
