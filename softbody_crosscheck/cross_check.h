// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: sanitizer declaration
//

#pragma once

#include <softbody.h>
#include <exception>

class Cross_Check_Listener {
public:
    virtual ~Cross_Check_Listener() = default;

    void fault(
            sb::Compute_Preference backend,
            sb::index_t pidx,
            sb::Particle ref,
            sb::Particle other,
            char const* message,
            char const* step);

protected:
    virtual void on_fault(
            sb::Compute_Preference backend,
            sb::index_t pidx,
            sb::Particle ref,
            sb::Particle other,
            char const* message,
            char const* step) {}
};

class Cross_Check {
public:
    Cross_Check();

    void step(Cross_Check_Listener* l);
    void dump_images(sb::ISerializer* serializers[3]);
    
    struct Simulation_Instance {
        sb::Unique_Ptr<sb::ISoftbody_Simulation> sim;
        sb::Unique_Ptr<sb::ISingle_Step_State> step;
    };

private:
    Simulation_Instance simulations[3];
};
