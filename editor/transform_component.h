// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#pragma once

#include <glm/vec3.hpp>
#include <glm/gtc/quaternion.hpp>

struct Transform_Component {
    glm::vec3 position;
    glm::quat rotation;
    glm::vec3 scale;

    bool manipulated = false;
};
