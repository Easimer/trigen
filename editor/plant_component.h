// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#pragma once

#include <memory>
#include <softbody.h>

struct Plant_Component {
	Plant_Component(sb::Config const &cfg) {
		_sim = sb::create_simulation(cfg);
	}

	sb::Unique_Ptr<sb::ISoftbody_Simulation> _sim;
};