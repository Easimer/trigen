// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: softbody simulation library
//

struct Softbody_Simulation;

namespace sb {
    struct config {
    };

    Softbody_Simulation* create_simulation(config const& configuration);
    void destroy_simulation(Softbody_Simulation*);
}
