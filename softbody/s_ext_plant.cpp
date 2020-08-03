// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: plant simulation
//

#include "stdafx.h"
#include "common.h"
#include "softbody.h"
#include "s_ext.h"
#include "m_sdf.h"

class Plant_Simulation : public ISimulation_Extension {
    void post_prediction(IParticle_Manager* pman, System_State& s) override {
        auto const anchor_point_min_dist = 2.0f;
        auto const attachment_strength = 0.25f;
        for (index_t i = 0; i < s.position.size(); i++) {
            Vector<Vec3> anchor_points;

            auto p = s.predicted_position[i];
            for (auto& C : s.colliders_sdf) {
                auto t = C.fun(p);
                if (t < anchor_point_min_dist) {
                    auto normal = sdf::normal(C.fun, p);
                    auto anchor_point = p - t * normal;

                    s.predicted_position[i] += attachment_strength * (anchor_point - s.predicted_position[i]);
                }
            }
        }
    }
};

sb::Unique_Ptr<ISimulation_Extension> Create_Extension_Plant_Simulation(sb::Extension kind) {
    return std::make_unique<Plant_Simulation>();
}
