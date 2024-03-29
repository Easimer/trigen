// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: serialization and deserialization internals
//

#pragma once

#include <cstdint>
#include "system_state.h"

#define MAKE_4BYTE_ID_X(c0, c1, c2, c3) \
    (uint32_t)((((c0) & 0xFF) << 24) | (((c1) & 0xFF) << 16) | (((c2) & 0xFF) << 8) | ((c3) & 0xFF))

#define MAKE_4BYTE_ID(c0, c1, c2, c3) MAKE_4BYTE_ID_X((unsigned char)(c0), (unsigned char)(c1), (unsigned char)(c2), (unsigned char)(c3))

#define IMAGE_MAGIC0 MAKE_4BYTE_ID('E', 'A', 'S', 'I')
#define IMAGE_MAGIC1 MAKE_4BYTE_ID('s', 'S', 'I', 'M')
#define IMAGE_VERSION (1)

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
std::enable_if_t<
    std::is_standard_layout_v<T>
> serialize(sb::ISerializer* serializer, Vector<T> const& v, u32 id) {
    u32 count = v.size();
    // Write chunk id
    serializer->write(&id, sizeof(id));
    // Write particle count
    serializer->write(&count, sizeof(count));
    // Write particle data
    serializer->write(v.data(), count * sizeof(T));
}

template<typename T>
std::enable_if_t<
    std::is_standard_layout_v<T>
> deserialize(sb::IDeserializer* deserializer, Vector<T>& v) {
    u32 count;
    // Read particle count
    deserializer->read(&count, sizeof(count));
    // Write particle data
    v.resize(count);
    deserializer->read(v.data(), count * sizeof(T));
}

void serialize(sb::ISerializer* serializer, Map<index_t, Vector<index_t>> const& m, u32 id);
void deserialize(sb::IDeserializer* deserializer, Map<index_t, Vector<index_t>>& m);

template<typename K, typename V>
std::enable_if_t<
    std::is_standard_layout_v<K> &&
    std::is_standard_layout_v<V>
> serialize(sb::ISerializer* serializer, Map<K, V> const& m, u32 id) {
    u32 count = m.size();

    // Write chunk id
    serializer->write(&id, sizeof(id));
    // Write particle count
    serializer->write(&count, sizeof(count));

    for (auto& kv : m) {
        auto const& first = kv.first;
        auto const& second = kv.second;
        serializer->write(&first, sizeof(first));
        serializer->write(&second, sizeof(second));
    }
}

template<typename K, typename V>
std::enable_if_t<
    std::is_standard_layout_v<K> &&
    std::is_standard_layout_v<V>
> deserialize(sb::IDeserializer* deserializer, Map<K, V>& m) {
    m.clear();

    u32 count;
    deserializer->read(&count, sizeof(count));

    for (u32 i = 0; i < count; i++) {
        K k;
        V v;

        deserializer->read(&k, sizeof(k));
        deserializer->read(&v, sizeof(v));

        m[k] = v;
    }
}

void deserialize_dispatch(sb::IDeserializer* deserializer, System_State& s, u32 id);
