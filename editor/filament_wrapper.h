// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: RAII wrapper for Filament resources
//

#pragma once

#include <filament/Engine.h>

namespace filament {
	template<typename T>
	class Ptr {
	public:
		Ptr() : _ptr(nullptr) {
		}

		Ptr(T ptr, filament::Engine *engine) : _ptr(ptr), _engine(engine) {
		}

		~Ptr() {
			reset();
		}

		T *operator->() {
			return _ptr;
		}

		operator T*() {
			return _ptr;
		}

		void reset() {
			if (_engine != nullptr) {
				_engine->destroy(_ptr);
			}
		}
	private:
		T *_ptr;
		filament::Engine *_engine;
	};
}
