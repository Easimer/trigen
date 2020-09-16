// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: fault detector declaration
//

#pragma once

#include <softbody.h>
#include <exception>

class Fault_Detector_Exception : public std::exception {
public:
	using Backend = sb::Compute_Preference;

	Fault_Detector_Exception(
		Backend backend,
		sb::index_t pidx,
		sb::Particle const& p0, sb::Particle const& p1,
		std::string const& message,
		std::string const& step) :
		_backend(backend),
		_pidx(pidx),
		_p0(p0), _p1(p1),
		_message(message),
		_step(step) {
	}

	Backend backend() const {
		return _backend;
	}

	sb::index_t which() const {
		return _pidx;
	}

	char const* step() const {
		return _step.c_str();
	}

	char const* what() const override {
		return _message.c_str();
	}

	sb::Particle const& particle0() const {
		return _p0;
	}

	sb::Particle const& particle1() const {
		return _p1;
	}
	
private:
	Backend _backend;
	sb::index_t _pidx;
	sb::Particle _p0, _p1;
	std::string _message;
	std::string _step;
};

class Fault_Detector {
public:
	Fault_Detector();

	void step();
	
	struct Simulation_Instance {
		sb::Unique_Ptr<sb::ISoftbody_Simulation> sim;
		sb::Unique_Ptr<sb::ISingle_Step_State> step;
	};

private:
	Simulation_Instance simulations[3];
};