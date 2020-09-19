// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: sanitizer declaration
//

#pragma once

#include <softbody.h>
#include <exception>

class Cross_Check {
public:
	Cross_Check();

	void step();
	
	struct Simulation_Instance {
		sb::Unique_Ptr<sb::ISoftbody_Simulation> sim;
		sb::Unique_Ptr<sb::ISingle_Step_State> step;
	};

private:
	Simulation_Instance simulations[3];
};
