// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: 
//

#pragma once

#include <cstddef>
#include <memory>
#include <optional>

#include <glm/vec2.hpp>

#include <psp/psp.h>

class UVSpatialIndex {
public:
    virtual ~UVSpatialIndex() = default;

    virtual std::optional<size_t> find_triangle(glm::vec2 const &uv) = 0;
};

std::unique_ptr<UVSpatialIndex> make_uv_spatial_index(PSP::Mesh const *mesh);
