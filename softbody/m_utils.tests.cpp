// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: m_utils tests
//

#include "stdafx.h"
#include <catch2/catch.hpp>
#include "m_utils.h"

TEST_CASE("longest_axis_normalized") {
    Vec3 tests[] = {
        { 4, 1, 2 }, { 1, 7, 2 }, { 0, 0, 10 },
    };

    Vec3 expected[] = {
        { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 },
    };

    for(int i = 0; i < 3; i++) {
        auto y = longest_axis_normalized(tests[i]);
        REQUIRE(y == expected[i]);
    }
}
