// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#include "stdafx.h"
#include <trigen/mesh_compress.h>

HEDLEY_BEGIN_C_DECLS

TMC_API
ETMC_Status
TMC_CreateContext(TMC_Context* contextPtr) {
    return k_ETMCStatus_Failure;
}

TMC_API
ETMC_Status
TMC_DestroyContext(TMC_Context context) {
    return k_ETMCStatus_Failure;
}

TMC_API
ETMC_Status
TMC_SetIndexArrayType(TMC_Context context, ETMC_Type type) {
    return k_ETMCStatus_Failure;
}

TMC_API
ETMC_Status
TMC_CreateBuffer(TMC_Context context, TMC_Buffer* buffer, const void* data, TMC_Size size) {
    return k_ETMCStatus_Failure;
}

TMC_API
ETMC_Status
TMC_CreateAttribute(TMC_Context context, TMC_Attribute* attribute, TMC_Buffer buffer, int numComponents, ETMC_Type type, TMC_Size stride, TMC_Size offset) {
    return k_ETMCStatus_Failure;
}

TMC_API
ETMC_Status
TMC_Compress(TMC_Context context) {
    return k_ETMCStatus_Failure;
}

TMC_API
ETMC_Status
TMC_GetDirectArray(TMC_Context context, TMC_Attribute attribute, const void** data, TMC_Size* size) {
    return k_ETMCStatus_Failure;
}

TMC_API
ETMC_Status
TMC_GetIndexArray(TMC_Context context, const void** data, TMC_Size* size) {
    return k_ETMCStatus_Failure;
}

HEDLEY_END_C_DECLS