// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: cloth demo
//

#include "stdafx.h"
#include "system_state.h"
#include "softbody.h"
#include "s_ext.h"
#include "m_utils.h"

class Cloth_Demo : public ISimulation_Extension {
public:
    Cloth_Demo(sb::Config const& params) : params(params) {
        extra.dim = 64;

        if(params.extra.cloth_sim != nullptr) {
            extra = *params.extra.cloth_sim;
        }
    }
private:
    sb::Config params;
    sb::Debug_Cloth_Extension_Extra extra;

    void init(IParticle_Manager_Deferred* pman, System_State& s, float dt) override {
        pman->defer([&](IParticle_Manager* pman, System_State&) {
            auto const N = extra.dim;
            auto const N_half = N / 2;
            for (int y = 0; y < N; y++) {
                for (int x = 0; x < N; x++) {
                    auto pos = Vec3(x - N_half, 16, y - N_half);
                    auto cur = pman->add_init_particle(pos, Vec3(0.25, 0.25, 1), 1);

                    if (x != 0) {
                        pman->connect_particles(cur - 1, cur);
                    }

                    if (y != 0) {
                        pman->connect_particles(cur - N, cur);
                        if (x != 0) {
                            pman->connect_particles(cur - N - 1, cur);
                        }
                    }
                }
            }

            index_t fixed[2] = { 0, N - 1 };
            pman->add_fixed_constraint(2, fixed);
        });
    }
};

sb::Unique_Ptr<ISimulation_Extension> Create_Extension_Cloth_Demo(sb::Extension kind, sb::Config const& params) {
    return std::make_unique<Cloth_Demo>(params);
}
