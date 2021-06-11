// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: structs
//

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <trigen/mesh_compress.h>

#include "arena.h"

using TMC_Index = std::make_signed<size_t>::type;

struct TMC_Buffer_t {
    std::unique_ptr<uint8_t[]> data;
    TMC_Size size;
};

struct TMC_Attribute_t {
    ETMC_Type type;
    unsigned numComponents;
    TMC_Size stride, offset;
    TMC_Buffer buffer;

    Arena compressedBuf;
};

struct TMC_Context_t {
    TMC_Bitfield hints;
    TMC_Size windowSize = 0;

    std::vector<std::unique_ptr<TMC_Buffer_t>> buffers;
    std::vector<std::unique_ptr<TMC_Attribute_t>> attributes;

    ETMC_Type indexType = ETMC_Type::k_ETMCType_UInt32;
    std::unique_ptr<uint8_t[]> indexBuffer;
    TMC_Size indexBufferSize = 0;
    TMC_Size indexBufferCount = 0;

    TMC_Index numOutputVertices = 0;

    TMC_Debug_Message_Proc dbgMsg = nullptr;
    void *dbgMsgUser = nullptr;
};

