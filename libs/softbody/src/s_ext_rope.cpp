// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: rope demo
//

#include "stdafx.h"
#include "softbody.h"
#include "s_ext.h"

#if SOFTBODY_BUILD_DEMO_EXTENSIONS

#include "system_state.h"
#include "m_utils.h"
#include <functional>

class Rope_Demo : public ISimulation_Extension {
public:
    Rope_Demo(sb::Config const& params) : params(params) {
    }
private:
    sb::Config params;

    void init(IParticle_Manager_Deferred* pman, System_State& s, float dt) override {
        pman->defer([&](IParticle_Manager* pman, System_State&) {
            make_curve(pman, 16, Vec3(0, 0, 0), std::bind(&curve_straight, std::placeholders::_1, Vec3(1, 0, 0)));
            make_curve(pman, 16, Vec3(0, 0, 5), std::bind(&curve_straight, std::placeholders::_1, Vec3(0, 1, 0)));
        });
    }

    static Vec3 curve_straight(int i, Vec3 const& dir) {
        return (float)i * dir;
    }

    void make_curve(IParticle_Manager *pman, int N, Vec3 const& origin, std::function<Vec3(int)> const& curve) {
        index_t start, prev, cur;
        start = prev = pman->add_init_particle(origin + curve(0), Vec3(0.25, 0.25, 1), 1);

        for (int i = 1; i < N; i++) {
            cur = pman->add_init_particle(origin + curve(i), Vec3(0.25, 0.25, 1), 1);

            pman->connect_particles(prev, cur);
            prev = cur;
        }

        index_t fixed[1] = { start };
        pman->add_fixed_constraint(1, fixed);
    }
};

sb::Unique_Ptr<ISimulation_Extension> Create_Extension_Rope_Demo(sb::Extension kind, sb::Config const& params) {
    return std::make_unique<Rope_Demo>(params);
}

#else // SOFTBODY_BUILD_DEMO_EXTENSIONS

sb::Unique_Ptr<ISimulation_Extension> Create_Extension_Rope_Demo(sb::Extension kind, sb::Config const& params) {
    return nullptr;
}

#endif // SOFTBODY_BUILD_DEMO_EXTENSIONS
