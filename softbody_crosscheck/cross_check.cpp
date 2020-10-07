// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: cross-checker
//

#include "stdafx.h"
#include <softbody.h>
#include <algorithm>
#include "cross_check.h"

#define SIM_REF (0)
#define SIM_OCL (1)
#define SIM_PRP (2)

void Cross_Check_Listener::fault(
            sb::Compute_Preference backend,
            sb::index_t pidx,
            sb::Particle ref,
            sb::Particle other,
            char const* message,
            char const* step) {
    on_fault(backend, pidx, ref, other, message, step);
}

static Cross_Check::Simulation_Instance create_sim_instance(sb::Compute_Preference pref) {
    Cross_Check::Simulation_Instance ret;
    sb::Config cfg;
    cfg.ext = sb::Extension::Debug_Cloth;
    cfg.compute_preference = pref;
    cfg.particle_count_limit = 65536;

    ret.sim = sb::create_simulation(cfg);
    ret.step = ret.sim->begin_single_step();

    return ret;
}

Cross_Check::Cross_Check() {
    simulations[SIM_REF] = create_sim_instance(sb::Compute_Preference::Reference);
    simulations[SIM_OCL] = create_sim_instance(sb::Compute_Preference::GPU_OpenCL);
    simulations[SIM_PRP] = create_sim_instance(sb::Compute_Preference::GPU_Proprietary);
}

static std::map<sb::index_t, sb::Particle> gather_particles(sb::Unique_Ptr<sb::Particle_Iterator> const& it) {
    std::map<sb::index_t, sb::Particle> ret;

    for (; !it->ended(); it->step()) {
        auto p = it->get();
        ret.emplace(std::make_pair(p.id, p));
    }

    return ret;
}

static void compare_states(Cross_Check_Listener* listener, Cross_Check::Simulation_Instance* simulations, char const* what, std::map<sb::index_t, sb::Particle>& particles_ref, std::map<sb::index_t, sb::Particle>& particles_ocl, std::map<sb::index_t, sb::Particle>& particles_prp) {
    char msgbuf[512];
    char stepbuf[128];

    auto it_max = std::max_element(particles_ref.begin(), particles_ref.end(), [&](auto lhs, auto rhs) {
        return lhs.first < rhs.first;
    });
    assert(it_max != particles_ref.end() && "Is the simulation empty?");

    auto idx_max = it_max->first;

    auto const epsilon = 1.0f;

    for (sb::index_t i = 0; i < idx_max; i++) {
        auto p_ref = particles_ref[i];
        auto p_ocl = particles_ocl[i];
        auto p_prp = particles_prp[i];

        auto eq0 = epsilonEqual(p_ref.position, p_ocl.position, epsilon);
        if (!eq0[0] || !eq0[1] || !eq0[2]) {
            simulations[SIM_OCL].step->get_state_description(128, stepbuf);
            snprintf(msgbuf, 511, "'%s' don't match between reference and OpenCL implementations!", what);
            listener->fault(
                    sb::Compute_Preference::GPU_OpenCL,
                    i, p_ref, p_ocl,
                    msgbuf,
                    stepbuf);
        }

        auto eq1 = epsilonEqual(p_ref.position, p_prp.position, epsilon);
        if (!eq1[0] || !eq1[1] || !eq1[2]) {
            simulations[SIM_PRP].step->get_state_description(128, stepbuf);
            snprintf(msgbuf, 511, "'%s' don't match between reference and proprietary implementations!", what);
            listener->fault(
                    sb::Compute_Preference::GPU_Proprietary,
                    i, p_ref, p_prp,
                    msgbuf,
                    stepbuf);
        }
    }
}

#define GATHER_PARTICLES_AND_COMPARE(what, kind) \
{ \
        auto particles_ref = gather_particles(simulations[SIM_REF].sim->kind()); \
        auto particles_ocl = gather_particles(simulations[SIM_OCL].sim->kind()); \
        auto particles_prp = gather_particles(simulations[SIM_PRP].sim->kind()); \
        compare_states(listener, simulations, what, particles_ref, particles_ocl, particles_prp); \
}

void Cross_Check::step(Cross_Check_Listener* listener) {
    step_counter++;
    printf("Step #%zu\n", step_counter);
    simulations[SIM_REF].step->step();
    simulations[SIM_OCL].step->step();
    simulations[SIM_PRP].step->step();

    GATHER_PARTICLES_AND_COMPARE("Centers-of-masses", get_centers_of_masses);
    GATHER_PARTICLES_AND_COMPARE("Predicted positions", get_particles_with_predicted_positions);
}

void Cross_Check::dump_images(sb::ISerializer* serializers[3]) {
    for(int i = 0; i < 3; i++) {
        simulations[i].sim->save_image(serializers[i]);
    }
}
