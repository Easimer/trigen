// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#pragma once

#include <memory>
#include <trigen.hpp>

struct Plant_Component {
	Plant_Component(trigen::Session *session) : session(session) {
	}

	trigen::Session *session;
};