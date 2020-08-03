// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: random number generation
//

#pragma once
#include <random>

class Rand_Float {
public:
    Rand_Float()
        : rd(), el(rd()),
        uniform01(0, 1), uniform11(-1, 1) {}

    /**
     * Generate a random float between 0 and 1.
     */
    float normal() {
        return uniform01(el);
    }

    /**
     * Generate a random float between -1 and 1.
     */
    float central() {
        return uniform11(el);
    }

private:
    std::random_device rd;
    std::mt19937 el;
    std::uniform_real_distribution<float> uniform01;
    std::uniform_real_distribution<float> uniform11;
};
