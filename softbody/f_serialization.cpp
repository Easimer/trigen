// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: serialization and deserialization
//

#include "stdafx.h"
#include <cstdint>
#include <cstdlib>
#include "f_serialization.h"
#include "f_serialization.internal.h"

#include "s_ext.h"

void serialize(sb::ISerializer* serializer, Map<index_t, Vector<index_t>> const& m, u32 id) {
    u32 count = m.size();

    // Write chunk id
    serializer->write(&id, sizeof(id));
    // Write particle count
    serializer->write(&count, sizeof(count));

    // For all particles in m
    for (auto& kv : m) {
        // Write particle id
        u64 pid = kv.first;
        serializer->write(&pid, sizeof(pid));
        // Write neighbor count
        u64 n = kv.second.size();
        serializer->write(&n, sizeof(n));
        // Write neighbor IDs
        for(auto pidx : kv.second) {
            auto pidx64 = (u64)pidx;
            serializer->write(&pidx64, sizeof(pidx64));
        }
    }
}

void deserialize(sb::IDeserializer* deserializer, Map<index_t, Vector<index_t>>& m) {
    u32 count;
    deserializer->read(&count, sizeof(count));

    for (u32 i = 0; i < count; i++) {
        u64 pid, n;

        deserializer->read(&pid, sizeof(pid));
        deserializer->read(&n, sizeof(n));
        Vector<index_t> neighbors;
        neighbors.reserve(n);
        for(u64 j = 0; j < n; j++) {
            u64 pidx64;
            deserializer->read(&pidx64, sizeof(pidx64));
            neighbors.push_back(pidx64);
        }

        m[pid] = std::move(neighbors);
    }
}

void deserialize_dispatch(Softbody_Simulation* sim, sb::IDeserializer* deserializer, System_State& s, u32 id, sb::Config const& params) noexcept {
    switch (id) {
    case CHUNK_BIND_POSITION:     
        deserialize(deserializer, s.bind_pose);
        break;
    case CHUNK_CURR_POSITION:
        deserialize(deserializer, s.position);
        break;
    case CHUNK_VELOCITY:
        deserialize(deserializer, s.velocity);
        break;
    case CHUNK_ANG_VELOCITY:
        deserialize(deserializer, s.angular_velocity);
        break;
    case CHUNK_SIZE:
        deserialize(deserializer, s.size);
        break;
    case CHUNK_ORIENT:
        deserialize(deserializer, s.orientation);
        break;
    case CHUNK_DENSITY:
        deserialize(deserializer, s.density);
        break;
    case CHUNK_EDGES:
        deserialize(deserializer, s.edges);
        break;
    default:
        try {
            auto ext_kind = Extension_Lookup_Extension_Kind(id);
            auto ext = sim->create_extension(ext_kind, params);
            ext->load_image(deserializer, s);
        } catch (std::range_error const&) {
            fprintf(stderr, "sb: UNKNOWN CHUCK ID %x\n", id);
            abort();
        }
        break;
    }
}

bool sim_save_image(System_State const& s, sb::ISerializer* serializer, ISimulation_Extension* ext) {
    Image_Header const hdr = { {IMAGE_MAGIC0, IMAGE_MAGIC1}, IMAGE_VERSION, 0 };
    serializer->write(&hdr, sizeof(hdr));
    serialize(serializer, s.bind_pose, CHUNK_BIND_POSITION);
    serialize(serializer, s.position, CHUNK_CURR_POSITION);
    serialize(serializer, s.velocity, CHUNK_VELOCITY);
    serialize(serializer, s.angular_velocity, CHUNK_ANG_VELOCITY);
    serialize(serializer, s.size, CHUNK_SIZE);
    serialize(serializer, s.orientation, CHUNK_ORIENT);
    serialize(serializer, s.density, CHUNK_DENSITY);
    serialize(serializer, s.edges, CHUNK_EDGES);

    if (ext != nullptr && ext->wants_to_serialize()) {
        if (!ext->save_image(serializer, s)) {
            return false;
        }
    }

    return true;
}

Serialization_Result sim_load_image(Softbody_Simulation* sim, System_State& s, sb::IDeserializer* deserializer, sb::Config const& config) {
    Image_Header hdr;
    deserializer->read(&hdr, sizeof(hdr));

    if (hdr.magic[0] == IMAGE_MAGIC0 && hdr.magic[1] == IMAGE_MAGIC1) {
        if (hdr.version != IMAGE_VERSION) {
            return Serialization_Result::Bad_Version;
        }

        u32 id;
        size_t res;

        for(;;) {
            res = deserializer->read(&id, sizeof(id));
            if (res != 0) {
                deserialize_dispatch(sim, deserializer, s, id, config);
            } else {
                break;
            }
        }

        auto const N = s.bind_pose.size();

        s.predicted_position.resize(N);
        s.predicted_orientation.resize(N);
        s.bind_pose_center_of_mass.resize(N);
        s.bind_pose_inverse_bind_pose.resize(N);
        s.center_of_mass.resize(N);
        s.goal_position.resize(N);

        return Serialization_Result::OK;
    } else {
        return Serialization_Result::Bad_Format;
    }
}
