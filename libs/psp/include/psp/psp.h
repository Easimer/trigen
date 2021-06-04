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

    struct [[deprecated]] Texture {
        int width, height;
        void const *buffer;
    };

    struct [[deprecated]] Material {
        Texture base;
        Texture normal;
        Texture height;
        Texture roughness;
        Texture ao;
    };

    class [[deprecated]] IPainter {
    public:
        virtual ~IPainter() = default;

        virtual void step_painting(float dt) = 0;
        virtual bool is_painting_done() = 0;

        virtual size_t num_particles() = 0;
        virtual bool get_particle(size_t id, glm::vec3 *out_position, glm::vec3 *out_next_position, glm::vec3 *out_velocity) = 0;

        virtual void result(Material *out_material) = 0;
    };

    int unwrap(/* inout */ Mesh &mesh);

    struct [[deprecated]] Parameters {
        PSP::Mesh const *mesh;
        PSP::Material const *material;

        int out_width;
        int out_height;

        int subdiv_theta;
        int subdiv_phi;
    };

    [[deprecated]]
    std::unique_ptr<PSP::IPainter> make_painter(Parameters const &params);

    // ==========================
    // NEW API
    // ==========================

    /** \brief An input texture descriptor. Doesn't own the buffer. */ 
    struct Input_Texture {
        unsigned width, height;
        void const *buffer;
    };

    /** \brief A list of input textures. */
    using Input_Material = std::vector<Input_Texture>;

    /** \brief An output textures. */
    struct Output_Texture {
        unsigned width, height;
        std::unique_ptr<uint8_t[]> buffer;
    };

    /** \brief A list of output textures. */
    using Output_Material = std::vector<Output_Texture>;

    /** \brief Painting parameters. */
    struct Paint_Input {
        /** Pointer to the mesh; can't be NULL. */
        PSP::Mesh const *mesh;
        /** Reference to the input material */
        Input_Material const &inputMaterial;

        /** Width of the output textures. Can't be zero. */
        unsigned outWidth;
        /** Height of the output textures. Can't be zero. */
        unsigned outHeight;
    };

    /**
     * Performs texture painting using the input textures provided.
     * Takes a mesh, a list of input materials and the desired output size and returns
     * a material (with the same number of textures as the input material).
     *
     * \param params Parameters
     * \return The output material.
     */
    Output_Material paint(Paint_Input const &params);
}
