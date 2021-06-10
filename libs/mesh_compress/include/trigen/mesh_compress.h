// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#ifndef TRIGEN_MESH_COMPRESS_H
#define TRIGEN_MESH_COMPRESS_H

#include <hedley.h>
#include <stdint.h>

#if defined(TMC_COMPILATION)
    #define TMC_API HEDLEY_PUBLIC
#else
    #define TMC_API HEDLEY_IMPORT
#endif /* defined(TMC_COMPILATION) */

typedef enum ETMC_Status {
    k_ETMCStatus_OK = 0,
    k_ETMCStatus_Failure,
    k_ETMCStatus_InvalidArguments,
    k_ETMCStatus_OutOfMemory,
    k_ETMCStatus_NotReady,
} ETMC_Status;

typedef enum ETMC_Type {
    k_ETMCType_Float32 = 0,
    k_ETMCType_UInt16,
    k_ETMCType_UInt32,
} ETMC_Type;

typedef struct TMC_Context_t *TMC_Context;
typedef struct TMC_Buffer_t *TMC_Buffer;
typedef struct TMC_Attribute_t *TMC_Attribute;
typedef uint32_t TMC_Size;
typedef int TMC_Bool;
typedef void (*TMC_Debug_Message_Proc)(void *user, char const *message, TMC_Bool is_error);

HEDLEY_BEGIN_C_DECLS

TMC_API
ETMC_Status
TMC_CreateContext(TMC_Context *contextPtr);

TMC_API
ETMC_Status
TMC_DestroyContext(TMC_Context context);

TMC_API
ETMC_Status
TMC_SetDebugMessageCallback(TMC_Context context, TMC_Debug_Message_Proc proc, void *user);

TMC_API
ETMC_Status
TMC_SetIndexArrayType(TMC_Context context, ETMC_Type type);

TMC_API
ETMC_Status
TMC_GetIndexArrayType(TMC_Context context, ETMC_Type *type);

TMC_API
ETMC_Status
TMC_CreateBuffer(TMC_Context context, TMC_Buffer *buffer, const void *data, TMC_Size size);

TMC_API
ETMC_Status
TMC_CreateAttribute(TMC_Context context, TMC_Attribute *attribute, TMC_Buffer buffer, unsigned numComponents, ETMC_Type type, TMC_Size stride, TMC_Size offset);

TMC_API
ETMC_Status
TMC_Compress(TMC_Context context, TMC_Size vertex_count);

TMC_API
ETMC_Status
TMC_GetDirectArray(TMC_Context context, TMC_Attribute attribute, const void **data, TMC_Size *size);

TMC_API
ETMC_Status
TMC_GetIndexArray(TMC_Context context, const void **data, TMC_Size *size, TMC_Size *element_count);

TMC_API
ETMC_Status
TMC_GetIndexArrayElementCount(TMC_Context context, TMC_Size *element_count);

HEDLEY_END_C_DECLS

#endif /* TRIGEN_MESH_COMPRESS_H */