// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: serialization and deserialization
//

#include "stdafx.h"
#include <cstdint>
#include "f_serialization.h"

#define MAKE_4BYTE_ID(c0, c1, c2, c3) \
    (uint32_t)((((c0) & 0xFF) << 24) | (((c1) & 0xFF) << 16) | (((c2) & 0xFF) << 8) | ((c3) & 0xFF))

#define IMAGE_MAGIC0 MAKE_4BYTE_ID('E', 'A', 'S', 'I')
#define IMAGE_MAGIC1 MAKE_4BYTE_ID('s', 'S', 'I', 'M')
#define IMAGE_VERSION (0)

using u8  = uint8_t;
using u32 = uint32_t;
using u64 = uint64_t;
using s32 = int32_t;
using s64 = int64_t;

#pragma pack(push, 1)
struct Image_Header {
    u32 magic[2];
    u32 version;
    u32 flags;
};
#pragma pack(pop)

static_assert(sizeof(Image_Header) == 8 + 4 + 4);

#define CHUNK_BIND_POSITION     MAKE_4BYTE_ID('B', 'P', 'O', 'S')
#define CHUNK_CURR_POSITION     MAKE_4BYTE_ID('C', 'P', 'O', 'S')
#define CHUNK_VELOCITY          MAKE_4BYTE_ID(' ', 'V', 'E', 'L')
#define CHUNK_ANG_VELOCITY      MAKE_4BYTE_ID('A', 'V', 'E', 'L')
#define CHUNK_SIZE              MAKE_4BYTE_ID('S', 'I', 'Z', 'E')
#define CHUNK_ORIENT            MAKE_4BYTE_ID(' ', 'Q', 'U', 'A')
#define CHUNK_DENSITY           MAKE_4BYTE_ID('D', 'E', 'N', 'S')
#define CHUNK_EDGES             MAKE_4BYTE_ID('E', 'D', 'G', 'E')

template<typename T>
static void serialize(sb::ISerializer* serializer, Vector<T> const& v, u32 id) {
    u32 count = v.size();
    // Write chunk id
    serializer->write(&id, sizeof(id));
    // Write particle count
    serializer->write(&count, sizeof(count));
    // Write particle data
    serializer->write(v.data(), count * sizeof(T));
}

template<typename T>
static void deserialize(sb::IDeserializer* deserializer, Vector<T>& v) {
    u32 count;
    // Read particle count
    deserializer->read(&count, sizeof(count));
    // Write particle data
    v.resize(count);
    deserializer->read(v.data(), count * sizeof(T));
}

static void serialize(sb::ISerializer* serializer, Map<unsigned, Vector<unsigned>> const& m, u32 id) {
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
        serializer->write(kv.second.data(), n * sizeof(unsigned));
    }
}

static void deserialize(sb::IDeserializer* deserializer, Map<unsigned, Vector<unsigned>>& m) {
    u32 count;
    deserializer->read(&count, sizeof(count));

    for (u32 i = 0; i < count; i++) {
        u64 pid, n;

        deserializer->read(&pid, sizeof(pid));
        deserializer->read(&n, sizeof(n));
        for (u64 j = 0; j < n; j++) {
            Vector<unsigned> neighbors;
            neighbors.resize(n);
            deserializer->read(neighbors.data(), n * sizeof(unsigned));
        }
    }
}

static void deserialize_dispatch(sb::IDeserializer* deserializer, System_State& s, u32 id) {
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
    }
}

bool sim_save_image(System_State const& s, sb::ISerializer* serializer) {
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

    return true;
}

Serialization_Result sim_load_image(System_State& s, sb::IDeserializer* deserializer) {
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
                deserialize_dispatch(deserializer, s, id);
            } else {
                break;
            }
        }
    } else {
        return Serialization_Result::Bad_Format;
    }
}
