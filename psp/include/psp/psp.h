// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: 
//

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

namespace PSP {
    struct Mesh {
        // Filled in by you
        std::vector<glm::vec3> position;
        std::vector<glm::vec3> normal;
        std::vector<size_t> elements;

        // Filled in by us and must be empty
        std::vector<glm::vec2> uv;
        std::vector<glm::u8vec3> chart_debug_color;
    };

    struct Texture {
        int width, height;
        void const *buffer;
    };

    struct Material {
        Texture base;
        Texture normal;
        Texture height;
        Texture roughness;
        Texture ao;
    };

    class IPainter {
    public:
        virtual ~IPainter() = default;

        virtual void step_painting(float dt) = 0;
        virtual bool is_painting_done() = 0;

        virtual size_t num_particles() = 0;
        virtual bool get_particle(size_t id, glm::vec3 *out_position, glm::vec3 *out_next_position, glm::vec3 *out_velocity) = 0;

        virtual void result(Material *out_material) = 0;
    };

    int unwrap(/* inout */ Mesh &mesh);

    struct Parameters {
        PSP::Mesh const *mesh;
        PSP::Material const *material;

        int out_width;
        int out_height;

        int subdiv_theta;
        int subdiv_phi;
    };

    std::unique_ptr<PSP::IPainter> make_painter(Parameters const &params);
}
