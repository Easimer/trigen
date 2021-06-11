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

#define TMC_VERSION_MAJOR    (1)
#define TMC_VERSION_MINOR    (0)
#define TMC_VERSION_REVISION (0)
#define TMC_VERSION                                                            \
    HEDLEY_VERSION_ENCODE(                                                     \
        TMC_VERSION_MAJOR, TMC_VERSION_MINOR, TMC_VERSION_REVISION)

#define TMC_VERSION_1_0 HEDLEY_VERSION_ENCODE(1, 0, 0)

#if !defined(TMC_TARGET_VERSION)
#define TMC_TARGET_VERSION TMC_VERSION
#endif

#if TMC_TARGET_VERSION < TMC_VERSION_1_0
#define TMC_AVAILABLE_SINCE_1_0 HEDLEY_UNAVAILABLE(1.0)
#else
#define TMC_AVAILABLE_SINCE_1_0
#endif /* TMC_TARGET_VERSION < 1.0.0 */

/*!
  \def TMC_IN
  The function will only read the pointer. The pointer must be non-NULL!
*/

/*!
  \def TMC_IN_OPT
  The function will only read the pointer. The pointer may be non-NULL.
*/

/*!
  \def TMC_OUT
  The function will only write the pointer. The pointer must be non-NULL!
*/

/*!
  \def TMC_INOUT
  The function will read and write the pointer. The pointer must be non-NULL!
*/

/*!
  \def TMC_HANDLE
  The parameter must be a valid handle and can't be NULL.
*/

/*!
  \def TMC_HANDLE_ACQUIRE
  If the function succeeds, the value at the pointer will be a valid handle.
*/

/*!
  \def TMC_HANDLE_RELEASE
  The handle will be released.
*/

/*!
  \def TMC_RETURN_CODE
  Status code. See \ref ETMC_Status.
*/

#if defined(HEDLEY_MSVC_VERSION)
#define TMC_IN _In_ _Pre_notnull_
#define TMC_IN_OPT _In_
#define TMC_OUT _Out_ _Pre_notnull_
#define TMC_OUT_OPT _Out_
#define TMC_INOUT _Inout_ _Pre_notnull_

#define TMC_HANDLE _Inout_ _Pre_notnull_
#define TMC_HANDLE_ACQUIRE _Outptr_ _Pre_notnull_
#define TMC_HANDLE_RELEASE _Post_invalid_ _Pre_notnull_

#define TMC_RETURN_CODE _Success_(return == k_ETMCStatus_OK) ETMC_Status
#elif defined(__clang__)
#define TMC_IN
#define TMC_IN_OPT
#define TMC_OUT
#define TMC_OUT_OPT
#define TMC_INOUT

#define TMC_HANDLE __attribute__((use_handle("tmc")))
#define TMC_HANDLE_ACQUIRE __attribute__((acquire_handle("tmc")))
#define TMC_HANDLE_RELEASE __attribute__((release_handle("tmc")))
#define TMC_RETURN_CODE ETMC_Status
#else
#define TMC_IN
#define TMC_IN_OPT
#define TMC_OUT
#define TMC_OUT_OPT
#define TMC_INOUT

#define TMC_HANDLE
#define TMC_HANDLE_ACQUIRE
#define TMC_HANDLE_RELEASE
#define TMC_RETURN_CODE ETMC_Status
#endif

typedef enum ETMC_Status {
    /** No error */
    k_ETMCStatus_OK = 0,
    /** General failure */
    k_ETMCStatus_Failure,
    /** One or more arguments are invalid */
    k_ETMCStatus_InvalidArguments,
    /** The compressor has run out of memory */
    k_ETMCStatus_OutOfMemory,
    /**
     * The results can't be returned to the caller because they are not yet
     * ready.
     */
    k_ETMCStatus_NotReady,
} ETMC_Status;

typedef enum ETMC_Type {
    /** Single-precision floating point value */
    k_ETMCType_Float32 = 0,
    /** Word-sized unsigned integer value */
    k_ETMCType_UInt16,
    /** Double word-sized unsigned integer value */
    k_ETMCType_UInt32,
} ETMC_Type;

typedef enum ETMC_Hint {
    /** Empty hint bitfield */
    k_ETMCHint_None = 0,
    /** Allow the compressor to automatically switch to a smaller index type
       than specified by a call to TMC_SetIndexArrayType; e.g. UInt32 -> UInt16
       if the vertex count was less than 65536. */
    k_ETMCHint_AllowSmallerIndices = 1 << 0,
} HEDLEY_FLAGS ETMC_Hint;

typedef enum ETMC_Param {
    /** Controls the size of the window used by the compressor. Default is 0,
       which means that the window is as big as possible.
       The value must be a non-negative integer. */
    k_ETMCParam_WindowSize = 0x1000,
} ETMC_Param;

typedef enum ETMC_Message_Level {
    /** Miscellaneous information message */
    k_ETMCMsgLevel_Info = 0,
    /** Error message */
    k_ETMCMsgLevel_Error,
} ETMC_Message_Level;

/** Context handle */
typedef struct TMC_Context_t *TMC_Context;
/** Buffer handle */
typedef struct TMC_Buffer_t *TMC_Buffer;
/** Attribute handle */
typedef struct TMC_Attribute_t *TMC_Attribute;
/** Size type */
typedef uint32_t TMC_Size;
/** Signed integer type */
typedef int32_t TMC_Int;
/** Bitfield type */
typedef uint32_t TMC_Bitfield;
/** Boolean type */
typedef int TMC_Bool;

/**
 * Signature of the debug message callback function.
 *
 * \param[inout] user User pointer
 * \param[in] message Pointer to the null-terminated string containing the message.
 * \param level Message level
 */
typedef void (*TMC_Debug_Message_Proc)(void *user, char const *message, ETMC_Message_Level level);

#define TMC_INVALID_CONTEXT_HANDLE ((TMC_Context)(void*)0)
#define TMC_INVALID_BUFFER_HANDLE ((TMC_Buffer)(void*)0)
#define TMC_INVALID_ATTRIBUTE_HANDLE ((TMC_Attribute)(void*)0)

HEDLEY_BEGIN_C_DECLS

/**
 * \brief Create a new context
 *
 * TMC_CreateContext() and TMC_DestroyContext() manage the compression context.
 *
 * \param[out] contextPtr Specifies a location where the context handle will be
 * placed
 * \param hints A bitfield with various hints for the compressor, see
 * ETMC_Hints
 * \returns ::k_ETMCStatus_InvalidArguments if \p contextPtr is NULL;
 * \returns ::k_ETMCStatus_OutOfMemory if the context cannot be allocated.
 * 
 * \note If a bit with an unspecified meaning is set to 1 in the hints bitfield
 * then the behavior of the function is undefined.
 */
TMC_API
TMC_AVAILABLE_SINCE_1_0
TMC_RETURN_CODE
TMC_CreateContext(
    TMC_HANDLE_ACQUIRE TMC_Context *contextPtr,
    TMC_Bitfield hints);

/**
 * \brief Destroy a context
 *
 * TMC_CreateContext() and TMC_DestroyContext() manage the compression context.
 *
 * \param[inout] context Context handle
 * \returns ::k_ETMCStatus_InvalidArguments if \p context is invalid.
 *
 * \note If the pointer passed to this function is not a context handle
 * allocated by a call to TMC_CreateContext() then the behavior of this
 * function is undefined.
 */
TMC_API
TMC_AVAILABLE_SINCE_1_0
TMC_RETURN_CODE
TMC_DestroyContext(
    TMC_HANDLE_RELEASE TMC_Context context);

/**
 * \brief Sets a parameter of integer type
 * 
 * TMC_SetParamInteger() sets a parameter to an integer value.
 *
 * \param[inout] context Context handle
 * \param param Name of the parameter
 * \param value New value of the parameter
 * \returns ::k_ETMCStatus_InvalidArguments if \p context is invalid, param can't
 * be an integer value or value is out-of-range for the specified parameter.
 */
TMC_API
TMC_AVAILABLE_SINCE_1_0
TMC_RETURN_CODE
TMC_SetParamInteger(
    TMC_HANDLE TMC_Context context,
    ETMC_Param param,
    TMC_Int value);

/**
 * \brief Sets the debug message callback
 * 
 * TMC_SetDebugMessageCallback() sets the debug message callback for a given
 * context. The debug message callback is called when the compressor wants to
 * print some kind of message to the user.
 *
 * \param[inout] context Context handle
 * \param[in] proc Pointer to the callback function
 * \param[in] user User pointer
 * \returns ::k_ETMCStatus_InvalidArguments if the \p context is invalid.
 * \note If \p proc is nullptr then the debug message callback is reset for the
 * context.
 */
TMC_API
TMC_AVAILABLE_SINCE_1_0
TMC_RETURN_CODE
TMC_SetDebugMessageCallback(
    TMC_HANDLE TMC_Context context,
    TMC_IN_OPT TMC_Debug_Message_Proc proc,
    TMC_IN_OPT void *user);

/**
 * \brief Sets the type of the element index.
 * 
 * TMC_SetIndexArrayType() sets the type of the index in the index buffer.
 * 
 * \param[inout] context Context handle
 * \param type Type of the index
 * \returns ::k_ETMCStatus_InvalidArguments if the \p context is invalid or if \p
 * type can't be used as a type in the index buffer.
 *
 * \note If you set the index type and call TMC_GetIndexArray(), without a call
 * to TMC_Compress() inbetween, then the results of the getter are undefined.
 * \note The default index type is ::k_ETMCType_UInt32 but don't rely on this
 * fact as it may change in future version. Either always set the index type
 * explicitly or check it using TMC_GetIndexArrayType() before using the
 * contents of the index buffer.
 */
TMC_API
TMC_AVAILABLE_SINCE_1_0
TMC_RETURN_CODE
TMC_SetIndexArrayType(
    TMC_HANDLE TMC_Context context,
    ETMC_Type type);

/**
 * \brief Gets the type of the element index.
 *
 * TMC_GetIndexArrayType() gets the type of the index in the index buffer.
 * 
 * \param[inout] context Context handle
 * \param[out] type Specifies a location where the name of the type will be
 * placed.
 * \returns ::k_EMTCStatus_InvalidArguments if \p context is invalid or \p type
 * is nullptr.
 */
TMC_API
TMC_AVAILABLE_SINCE_1_0
TMC_RETURN_CODE
TMC_GetIndexArrayType(
    TMC_HANDLE TMC_Context context,
    TMC_OUT ETMC_Type *type);

/**
 * \brief Creates a new buffer
 * 
 * TMC_CreateBuffer() creates a new buffer and returns it's handle.
 * Buffers may contain any kind of data.
 *
 * \param[inout] context Context handle
 * \param[out] buffer Specifies a location where the handle of the buffer will
 * be stored. Can't be nullptr.
 * \param[in] data Specifies the pointer to the data that will be copied into
 * the buffer; can't be nullptr.
 * \param size Specifies the size in bytes of the buffer's data store.
 *
 * \returns ::k_ETMCStatus_InvalidArguments if \p context is invalid, \p buffer
 * is nullptr, \p data is nullptr or \p size is zero;
 * \returns ::k_ETMCStatus_OutOfMemory if the compressor has run out of memory.
 * 
 * \note This is analogous to a call to glCreateBuffers() and glBufferData() in
 * OpenGL.
 */
TMC_API
TMC_AVAILABLE_SINCE_1_0
TMC_RETURN_CODE
TMC_CreateBuffer(
    TMC_HANDLE TMC_Context context,
    TMC_HANDLE_ACQUIRE TMC_Buffer *buffer,
    TMC_IN const void *data,
    TMC_Size size);

/**
 * \brief Defines a new vertex attribute.
 *
 * TMC_CreateAttribute() defines a new vertex attribute.
 * The mesh is understood to be made of triangles, not triangle strips or
 * triangle fans or any other kind of primitives.
 * 
 * \param [inout] context Context handle
 * \param [out] attribute Specifies the location where the handle of the
 * attribute will be stored. Can't be nullptr.
 * \param [in] buffer A valid handle to the buffer that this attribute's data
 * is contained in.
 * \param numComponents Number of components per vertex attribute. Must be 1,
 * 2, 3 or 4.
 * \param type The data type of each component.
 * Allowed types are:
 * - ::k_ETMCType_Float32
 * \param stride Byte offset between consecutive vertex attributes. Can't be 0.
 * \param offset Offset of the first component of the first vertex attribute in
 * the buffer.
 * \returns ::k_ETMCStatus_InvalidArguments if any of the arguments are
 * out-of-range or invalid.
 * \returns ::k_ETMCStatus_OutOfMemory if the compressor has run out of memory.
 */
TMC_API
TMC_AVAILABLE_SINCE_1_0
TMC_RETURN_CODE
TMC_CreateAttribute(
    TMC_HANDLE TMC_Context context,
    TMC_HANDLE_ACQUIRE TMC_Attribute *attribute,
    TMC_HANDLE TMC_Buffer buffer,
    unsigned numComponents,
    ETMC_Type type,
    TMC_Size stride,
    TMC_Size offset);

/**
 * \brief Compresses the mesh
 * 
 * TMC_Compress() performs the compression of the mesh.
 *
 * \param [inout] context Context handle
 * \param vertex_count Number of vertices in the uncompressed mesh.
 * \returns ::k_ETMCStatus_InvalidArguments if the context handle is invalid.
 * 
 * \note If any of the buffers contain less data than needed to build a
 * triangle mesh from \p vertex_count vertices then calling this function will
 * result in undefined behavior.
 *
 * \note The results of this operation will be available through
 * - TMC_GetDirectArray(),
 * - TMC_GetIndexArray() and
 * - TMC_GetIndexArrayElementCount().
 */
TMC_API
TMC_AVAILABLE_SINCE_1_0
TMC_RETURN_CODE
TMC_Compress(
    TMC_HANDLE TMC_Context context,
    TMC_Size vertex_count);

/**
 * \brief Returns the direct array of an attribute to the caller.
 *
 * TMC_GetDirectArray() returns the direct array of the specified attribute to the caller.
 *
 * \param [inout] context Context handle
 * \param [in] attribute Attribute handle
 * \param [out] data Specifies the location where the pointer to the vertex
 * attribute data is stored
 * \param [out] size Specifies the location where the size in bytes of the
 * vertex attribute data is stored
 * \returns ::k_ETMCStatus_InvalidArguments if any of the handles are invalid
 * or either data or size is nullptr.
 * \returns ::k_ETMCStatus_NotReady if TMC_Compress() wasn't called yet
 * 
 * \note Calling this function after modifying the context using
 * TMC_CreateBuffer(), TMC_CreateAttribute(), TMC_SetParamInteger() or
 * TMC_SetIndexArrayType() results in undefined behavior.
 */
TMC_API
TMC_AVAILABLE_SINCE_1_0
TMC_RETURN_CODE
TMC_GetDirectArray(
    TMC_HANDLE TMC_Context context,
    TMC_HANDLE TMC_Attribute attribute,
    TMC_OUT const void **data,
    TMC_OUT TMC_Size *size);

/**
 * \brief Returns the index array of the mesh to the caller
 *
 * TMC_GetIndexArray() returns the index array of the mesh to the caller.
 *
 * \param [inout] context Context handle
 * \param [in] attribute Attribute handle
 * \param [out] data Specifies the location where the pointer to the index
 * array is stored
 * \param [out] size Specifies the location where the size in bytes of the
 * index array is stored
 * \param [out] element_count Specifies the location where the number of
 * elements in the index array is stored. Can be nullptr.
 * \returns ::k_ETMCStatus_InvalidArguments if any of the handles are invalid
 * or either \p data or \p size is nullptr.
 * \returns ::k_ETMCStatus_NotReady if TMC_Compress() wasn't called yet
 *
 * \note Calling this function after modifying the context using
 * TMC_CreateBuffer(), TMC_CreateAttribute(), TMC_SetParamInteger() or
 * TMC_SetIndexArrayType() results in undefined behavior.
 */
TMC_API
TMC_AVAILABLE_SINCE_1_0
TMC_RETURN_CODE
TMC_GetIndexArray(
    TMC_HANDLE TMC_Context context,
    TMC_OUT const void **data,
    TMC_OUT_OPT TMC_Size *size,
    TMC_OUT_OPT TMC_Size *element_count);

/**
 * \brief Returns the number of elements in the index array to the caller.
 *
 * TMC_GetIndexArrayElementCount() returns the number of elements in the index array to the caller.
 *
 * \param [inout] context Context handle
 * \param [out] element_count Specifies the location where the number of
 * elements in the index array is stored. Can't be nullptr.
 * \returns ::k_ETMCStatus_InvalidArguments if the context handle is invalid or
 * \p element_count is nullptr.
 * \returns ::k_ETMCStatus_NotReady if TMC_Compress() wasn't called yet
 *
 * \note Calling this function after modifying the context using
 * TMC_CreateBuffer(), TMC_CreateAttribute(), TMC_SetParamInteger() or
 * TMC_SetIndexArrayType() results in undefined behavior.
 */
TMC_API
TMC_AVAILABLE_SINCE_1_0
TMC_RETURN_CODE
TMC_GetIndexArrayElementCount(
    TMC_HANDLE TMC_Context context,
    TMC_OUT TMC_Size *element_count);

HEDLEY_END_C_DECLS

#endif /* TRIGEN_MESH_COMPRESS_H */