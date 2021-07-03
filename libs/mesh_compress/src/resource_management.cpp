// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#include "stdafx.h"

#include "arena.h"
#include "dbgmsg.h"
#include <cstring>
#include <trigen/mesh_compress.h>

HEDLEY_BEGIN_C_DECLS

TMC_API
TMC_RETURN_CODE
TMC_CreateContext(
    TMC_HANDLE_ACQUIRE TMC_Context *contextPtr,
    TMC_Bitfield hints) {
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
TMC_RETURN_CODE
TMC_DestroyContext(
    TMC_HANDLE_RELEASE TMC_Context context) {
    if (context == nullptr) {
        return k_ETMCStatus_InvalidArguments;
    }

    delete context;

    return k_ETMCStatus_OK;
}

TMC_API
TMC_RETURN_CODE
TMC_SetIndexArrayType(
    TMC_HANDLE TMC_Context context,
    ETMC_Type type) {
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
TMC_RETURN_CODE
TMC_GetIndexArrayType(
    TMC_HANDLE TMC_Context context,
    TMC_OUT ETMC_Type *type) {
    if (context == nullptr || type == nullptr) {
        return k_ETMCStatus_InvalidArguments;
    }

    *type = context->indexType;

    return k_ETMCStatus_OK;
}

TMC_API
TMC_RETURN_CODE
TMC_CreateBuffer(
    TMC_HANDLE TMC_Context context,
    TMC_HANDLE_ACQUIRE TMC_Buffer *buffer,
    TMC_IN const void *data,
    TMC_Size size) {
    if (context == nullptr || buffer == nullptr || (data == nullptr && size != 0)) {
        return k_ETMCStatus_InvalidArguments;
    }

    auto bufData = std::make_unique<uint8_t[]>(size);
    if (bufData == nullptr) {
        return k_ETMCStatus_OutOfMemory;
    }

    memcpy(bufData.get(), data, size);
    auto buf = TMC_Buffer_t { std::move(bufData), size };
    auto bufPtr = std::make_unique<TMC_Buffer_t>(std::move(buf));
    if (bufPtr == nullptr) {
        return k_ETMCStatus_OutOfMemory;
    }

    *buffer = bufPtr.get();
    context->buffers.emplace_back(std::move(bufPtr));

    return k_ETMCStatus_OK;
}

TMC_API
TMC_RETURN_CODE
TMC_CreateAttribute(
    TMC_HANDLE TMC_Context context,
    TMC_HANDLE_ACQUIRE TMC_Attribute *attribute,
    TMC_HANDLE TMC_Buffer buffer,
    unsigned numComponents,
    ETMC_Type type,
    TMC_Size stride,
    TMC_Size offset) {
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
    if (attr == nullptr) {
        return k_ETMCStatus_OutOfMemory;
    }

    *attribute = attr.get();
    context->attributes.emplace_back(std::move(attr));

    return k_ETMCStatus_OK;
}

TMC_API
TMC_RETURN_CODE
TMC_GetDirectArray(
    TMC_HANDLE TMC_Context context,
    TMC_HANDLE TMC_Attribute attribute,
    TMC_OUT const void **data,
    TMC_OUT TMC_Size *size) {
    if (context == nullptr || attribute == nullptr || data == nullptr || size == nullptr) {
        return k_ETMCStatus_InvalidArguments;
    }

    if (context->indexBuffer == nullptr) {
        return k_ETMCStatus_NotReady;
    }

    *data = attribute->compressedBuf.data();
    *size = attribute->compressedBuf.size();

    return k_ETMCStatus_OK;
}

TMC_API
TMC_RETURN_CODE
TMC_GetIndexArray(
    TMC_HANDLE TMC_Context context,
    TMC_OUT const void **data,
    TMC_OUT_OPT TMC_Size *size,
    TMC_OUT_OPT TMC_Size *element_count) {
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
TMC_RETURN_CODE
TMC_GetIndexArrayElementCount(
    TMC_HANDLE TMC_Context context,
    TMC_OUT TMC_Size *element_count) {
    if (context == nullptr || element_count == nullptr) {
        return k_ETMCStatus_InvalidArguments;
    }

    if (context->indexBuffer == nullptr) {
        return k_ETMCStatus_NotReady;
    }

    *element_count = context->indexBufferCount;

    return k_ETMCStatus_OK;
}

TMC_API
TMC_RETURN_CODE
TMC_SetParamInteger(
    TMC_HANDLE TMC_Context context,
    ETMC_Param param,
    TMC_Int value) {
    if (context == nullptr) {
        return k_ETMCStatus_InvalidArguments;
    }

    switch (param) {
    case k_ETMCParam_WindowSize:
        if (value < 0) {
            TMC_PrintError(context,
                "TMC_SetParamInteger: k_ETMCParam_WindowSize cannot be less "
                "than zero!\n");
            return k_ETMCStatus_InvalidArguments;
        }

        context->windowSize = TMC_Size(value);
        break;
    default:
        TMC_PrintError(context,
            "TMC_SetParamInteger called with invalid parameter kind '%x'\n",
            param);
        return k_ETMCStatus_InvalidArguments;
    }

    return k_ETMCStatus_OK;
}

HEDLEY_END_C_DECLS