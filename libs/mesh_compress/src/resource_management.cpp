// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#include "stdafx.h"

#include "arena.h"
#include "dbgmsg.h"
#include <trigen/mesh_compress.h>

HEDLEY_BEGIN_C_DECLS

TMC_API
ETMC_Status
TMC_CreateContext(TMC_Context *contextPtr, TMC_Bitfield hints) {
    if (contextPtr == nullptr) {
        return k_ETMCStatus_InvalidArguments;
    }

    *contextPtr = new TMC_Context_t;

    if (*contextPtr == nullptr) {
        return k_ETMCStatus_OutOfMemory;
    }

    (*contextPtr)->hints = hints;

    return k_ETMCStatus_OK;
}

TMC_API
ETMC_Status
TMC_DestroyContext(TMC_Context context) {
    if (context == nullptr) {
        return k_ETMCStatus_InvalidArguments;
    }

    delete context;

    return k_ETMCStatus_OK;
}

TMC_API
ETMC_Status
TMC_SetIndexArrayType(TMC_Context context, ETMC_Type type) {
    if (context == nullptr) {
        return k_ETMCStatus_InvalidArguments;
    }

    if (type != k_ETMCType_UInt16 && type != k_ETMCType_UInt32) {
        TMC_PrintError(context, "Indices can't be of type '%d'!\n", type);
        return k_ETMCStatus_InvalidArguments;
    }

    if (context->indexType != type) {
        TMC_Print(context, "Index type has changed; resetting index buffer\n");
        context->indexBuffer.reset();
    }
    context->indexType = type;

    return k_ETMCStatus_OK;
}

TMC_API
ETMC_Status
TMC_GetIndexArrayType(TMC_Context context, ETMC_Type *type) {
    if (context == nullptr || type == nullptr) {
        return k_ETMCStatus_InvalidArguments;
    }

    *type = context->indexType;

    return k_ETMCStatus_OK;
}

TMC_API
ETMC_Status
TMC_CreateBuffer(TMC_Context context, TMC_Buffer *buffer, const void *data, TMC_Size size) {
    if (context == nullptr || buffer == nullptr || data == nullptr || size == 0) {
        return k_ETMCStatus_InvalidArguments;
    }

    auto bufData = std::make_unique<uint8_t[]>(size);
    memcpy(bufData.get(), data, size);
    auto buf = TMC_Buffer_t { std::move(bufData), size };
    auto bufPtr = std::make_unique<TMC_Buffer_t>(std::move(buf));
    *buffer = bufPtr.get();
    context->buffers.emplace_back(std::move(bufPtr));

    return k_ETMCStatus_OK;
}

TMC_API
ETMC_Status
TMC_CreateAttribute(TMC_Context context, TMC_Attribute *attribute, TMC_Buffer buffer, unsigned numComponents, ETMC_Type type, TMC_Size stride, TMC_Size offset) {
    if (context == nullptr || attribute == nullptr || buffer == nullptr || numComponents <= 0) {
        return k_ETMCStatus_InvalidArguments;
    }

    if (stride == 0) {
        TMC_PrintError(context, "Stride must be greater than zero!\n");
        return k_ETMCStatus_InvalidArguments;
    }

    if (numComponents > 4) {
        TMC_PrintError(context, "Number of components can't be greater than 4!\n");
        return k_ETMCStatus_InvalidArguments;
    }

    if (type != k_ETMCType_Float32) {
        TMC_PrintError(context, "Type of attribute can't be '%d'!\n", type);
        return k_ETMCStatus_InvalidArguments;
    }

    auto attr = std::make_unique<TMC_Attribute_t>(TMC_Attribute_t { type, numComponents, stride, offset, buffer });
    *attribute = attr.get();
    context->attributes.emplace_back(std::move(attr));

    return k_ETMCStatus_OK;
}

TMC_API
ETMC_Status
TMC_GetDirectArray(TMC_Context context, TMC_Attribute attribute, const void **data, TMC_Size *size) {
    if (context == nullptr || attribute == nullptr || data == nullptr || size == nullptr) {
        return k_ETMCStatus_InvalidArguments;
    }

    *data = attribute->compressedBuf.get();
    *size = attribute->compressedSize;

    return k_ETMCStatus_OK;
}

TMC_API
ETMC_Status
TMC_GetIndexArray(TMC_Context context, const void **data, TMC_Size *size, TMC_Size *element_count) {
    if (context == nullptr || data == nullptr || (size == nullptr && element_count == nullptr)) {
        return k_ETMCStatus_InvalidArguments;
    }

    if (context->indexBuffer == nullptr) {
        return k_ETMCStatus_NotReady;
    }

    *data = context->indexBuffer.get();

    if (size) {
        *size = context->indexBufferSize;
    }

    if (element_count) {
        *element_count = context->indexBufferCount;
    }

    return k_ETMCStatus_OK;
}

TMC_API
ETMC_Status
TMC_GetIndexArrayElementCount(TMC_Context context, TMC_Size *element_count) {
    if (context == nullptr || element_count == nullptr) {
        return k_ETMCStatus_InvalidArguments;
    }

    *element_count = context->indexBufferCount;

    return k_ETMCStatus_OK;
}

HEDLEY_END_C_DECLS