// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: c_particle tests
//

#include "stdafx.h"
#include <catch2/catch.hpp>
#include "c_particle.h"

TEST_CASE("Two spheres too far away") {
    Vec3 unit_size(0.5, 0.5, 0.5);
    Vec3 p0(-10, 0, 0);
    Vec3 p1( 10, 0, 0);
    auto normal = glm::normalize(p1 - p0);
    auto r_id = glm::identity<glm::quat>();
    auto res = coll::particle_particle_distance(
        normal,
        p0, r_id, unit_size,
        p1, r_id, unit_size
    );

    REQUIRE(!res.has_value());
}

TEST_CASE("Two spheres in the same space") {
    Vec3 unit_size(0.5, 0.5, 0.5);
    Vec3 p(10, 10, 10);
    auto normal = Vec3(1, 0, 0);
    auto r_id = glm::identity<glm::quat>();
    auto res = coll::particle_particle_distance(
        normal,
        p, r_id, unit_size,
        p, r_id, unit_size
    );

    REQUIRE(res.has_value());
    REQUIRE((1 < *res && *res < 2));
}

TEST_CASE("Two spheres overlapping") {
    Vec3 unit_size(0.5, 0.5, 0.5);
    Vec3 p0(0, 0, 0);
    Vec3 p1(0, 0.75, 0);
    auto normal = glm::normalize(p1 - p0);
    auto r_id = glm::identity<glm::quat>();
    auto res = coll::particle_particle_distance(
        normal,
        p0, r_id, unit_size,
        p1, r_id, unit_size
    );

    REQUIRE(res.has_value());
    REQUIRE(0 < *res);

}
