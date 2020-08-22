// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: simulation extensions
//

#pragma once

#include "common.h"
#include "softbody.h"
#include <array>
#include "l_constexpr_map.h"

class IParticle_Manager {
public:
    virtual index_t add_init_particle(Vec3 const& p_pos, Vec3 const& p_size, float p_density) = 0;
    virtual void connect_particles(index_t a, index_t b) = 0;

    virtual index_t add_particle(Vec3 const& p_pos, Vec3 const& p_size, float p_density, index_t parent) = 0;

    virtual void add_fixed_constraint(unsigned count, index_t* pidx) = 0;

    virtual void invalidate_particle_cache() = 0;
};

class IParticle_Manager_Deferred {
public:
    virtual void defer(std::function<void(IParticle_Manager* pman, System_State& s)> const& f) = 0;
};

class ISimulation_Extension {
public:
    virtual ~ISimulation_Extension() {}
#define SIMEXT_CALLBACK(name) virtual void name (IParticle_Manager_Deferred*, System_State&, float dt) {}
    SIMEXT_CALLBACK(init)
    SIMEXT_CALLBACK(pre_prediction)
    SIMEXT_CALLBACK(post_prediction)
    SIMEXT_CALLBACK(pre_constraint)
    SIMEXT_CALLBACK(post_constraint)
    SIMEXT_CALLBACK(pre_integration)
    SIMEXT_CALLBACK(post_integration)

    virtual bool wants_to_serialize() { return false; }
    virtual bool save_image(sb::ISerializer* serializer, System_State const& s) { return true; }
    virtual bool load_image(sb::IDeserializer* serializer, System_State& s) { return true; }
};

sb::Unique_Ptr<ISimulation_Extension> Create_Extension_Plant_Simulation(sb::Extension kind, sb::Config const& params);
sb::Unique_Ptr<ISimulation_Extension> Create_Extension_Cloth_Demo(sb::Extension kind, sb::Config const& params);

inline sb::Unique_Ptr<ISimulation_Extension> Create_Extension(sb::Extension kind, sb::Config const& params) {
    switch (kind) {
    case sb::Extension::Plant_Simulation: return Create_Extension_Plant_Simulation(kind, params);
    case sb::Extension::Debug_Cloth: return Create_Extension_Cloth_Demo(kind, params);
    default: return std::make_unique<ISimulation_Extension>();
    }
}

namespace detail {
    // TODO(danielm): duplicate of macro in f_serialization.internal.h!
    inline constexpr uint32_t make_4byte_id(char c0, char c1, char c2, char c3) {
        auto const u0 = static_cast<unsigned char>(c0);
        auto const u1 = static_cast<unsigned char>(c1);
        auto const u2 = static_cast<unsigned char>(c2);
        auto const u3 = static_cast<unsigned char>(c3);

        return (u0 << 24) | (u1 << 16) | (u2 << 8) | (u3 << 0);
    }

    using Ext_Chunk_Id_Map = Bijective_Constexpr_Map<sb::Extension, uint32_t, 4>;

    inline constexpr Ext_Chunk_Id_Map::Data_Source extension_chunk_ids {
        {
            { sb::Extension::None,              make_4byte_id('E', 'x', 'N', 'o') },
            { sb::Extension::Debug_Rope,        make_4byte_id('E', 'x', 'D', 'r') },
            { sb::Extension::Debug_Cloth,       make_4byte_id('E', 'x', 'D', 'c') },
            { sb::Extension::Plant_Simulation,  make_4byte_id('E', 'x', 'P', 'l') },
        }
    };
}

inline uint32_t Extension_Lookup_Chunk_Identifier(sb::Extension ext) {
    static constexpr auto map = detail::Ext_Chunk_Id_Map({ detail::extension_chunk_ids });

    return map.at(ext);
}

inline sb::Extension Extension_Lookup_Extension_Kind(uint32_t chunk_id) {
    static constexpr auto map = detail::Ext_Chunk_Id_Map({ detail::extension_chunk_ids });

    return map.at(chunk_id);
}
