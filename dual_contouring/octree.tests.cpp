// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: octree tests
//

#include <catch2/catch.hpp>
#include "octree.h"

TEST_CASE("Octree: empty tree") {
    Octree<4> tree(glm::vec3(0, 0, 0), glm::vec3(1, 1, 1));

    auto p = glm::vec3(0.82, -0.46, 0.31);
    REQUIRE(tree.count(p) == false);
}

TEST_CASE("Octree: add single point, check it") {
    Octree<4> tree(glm::vec3(0, 0, 0), glm::vec3(1, 1, 1));

    auto p = glm::vec3(0.82, -0.46, 0.31);
    tree.add_point(p);
    REQUIRE(tree.count(p) == true);
}

TEST_CASE("Octree: add single point, check other") {
    Octree<4> tree(glm::vec3(0, 0, 0), glm::vec3(1, 1, 1));

    auto p0 = glm::vec3(0.82, -0.46, 0.31);
    tree.add_point(p0);
    auto p1 = glm::vec3(-0.82, 0.46, -0.31);
    REQUIRE(tree.count(p1) == false);
}
