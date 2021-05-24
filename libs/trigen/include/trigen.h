// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: libtrigen functions
//

#ifndef H_EASI_TRIGEN
#define H_EASI_TRIGEN

#include <stdint.h>
#include <stddef.h>

#ifdef _WIN32
#define TRIGEN_EXPORT __declspec(dllexport)
#else
#define TRIGEN_EXPORT
#endif

#define TRIGEN_API TRIGEN_EXPORT __cdecl

#if _WIN32 && defined(_In_) && defined(_Out_) && defined(_Inout_) && defined(_Post_invalid_)
#define TRIGEN_IN _In_ _Pre_notnull_
#define TRIGEN_OUT _Out_
#define TRIGEN_INOUT _Inout_

#define TRIGEN_HANDLE _Inout_
#define TRIGEN_HANDLE_ACQUIRE _Outptr_
#define TRIGEN_HANDLE_RELEASE _Post_invalid_
#define TRIGEN_FREED _Post_invalid_
#elif defined(__clang__)
#define TRIGEN_IN
#define TRIGEN_OUT
#define TRIGEN_INOUT

#define TRIGEN_HANDLE __attribute__((use_handle("trigen")))
#define TRIGEN_HANDLE_ACQUIRE __attribute__((acquire_handle("trigen")))
#define TRIGEN_HANDLE_RELEASE __attribute__((release_handle("trigen")))
#define TRIGEN_FREED
#else
#define TRIGEN_IN
#define TRIGEN_OUT
#define TRIGEN_INOUT

#define TRIGEN_HANDLE
#define TRIGEN_HANDLE_ACQUIRE
#define TRIGEN_HANDLE_RELEASE
#define TRIGEN_FREED
#endif

typedef struct Trigen_Session_t* Trigen_Session;
typedef struct Trigen_Collider_t *Trigen_Collider;

typedef enum Trigen_Status_t {
    /** No error */
    Trigen_OK = 0,
    /** General failure */
    Trigen_Failure,
    /** One or more arguments have an invalid value */
    Trigen_InvalidArguments,
    /** The function has run out of heap space before it could finish */
    Trigen_OutOfMemory,
    /** The plant configuration structure contains one or more invalid values */
    Trigen_InvalidConfiguration,
    /** The mesh descriptor structure contains one or more invalid values */
    Trigen_InvalidMesh,
    /** A prerequisite input dataset hasn't been generated yet */
    Trigen_NotReady,
    /** Can't fit the data into the output array */
    Trigen_NotEnoughSpace,
} Trigen_Status;

#if _WIN32 && defined(_Check_return_)
#define TRIGEN_RETURN_CODE _Check_return_ Trigen_Status
#else
#if __cplusplus
#define TRIGEN_RETURN_CODE [[nodiscard]] Trigen_Status
#else
#define TRIGEN_RETURN_CODE Trigen_Status
#endif
#endif

enum Trigen_Flags {
    /** No flags */
    Trigen_F_None                   = 0,
    /** Prefer the CPU compute backend as opposed to the GPU ones */
    Trigen_F_PreferCPU              = 1 << 0,
    /** Ignored */
    Trigen_F_UseGeneralTexturingAPI = 1 << 1,
};

typedef struct Trigen_Parameters_t {
    unsigned int flags;

    float seed_position[3];

    float density;
    float attachment_strength;
    float surface_adaption_strength;
    float stiffness;
    float aging_rate;
    float phototropism_response_strength;
    float branching_probability;
    float branch_angle_variance;

    unsigned particle_count_limit;
} Trigen_Parameters;

typedef struct Trigen_Collider_Mesh_t {
    // Number of triangles in this mesh.
    // The number of elements in the index buffer should be three
    // times this number.
    size_t triangle_count;

    // Pointer to the vertex index buffer.
    uint64_t const *vertex_indices;
    // Pointer to the normal index buffer.
    uint64_t const *normal_indices;

    // Number of elements in the position buffer.
    // This should be at least `max(vertex_indices)+1`.
    size_t position_count;
    // Pointer to the position vector buffer.
    // Assumed to be in the following format: XYZ XYZ XYZ
    float const *positions;

    // Number of elements in the normal buffer.
    // This should be at least `max(normal_indices)+1`.
    size_t normal_count;
    // Pointer to the normal vector buffer.
    // Assumed to be in the following format: XYZ XYZ XYZ
    float const *normals;
} Trigen_Collider_Mesh;

typedef struct Trigen_Mesh_t {
    // Number of triangles in this mesh.
    // The number of elements in the index buffer should be three
    // times this number.
    size_t triangle_count;

    // Pointer to the vertex index buffer.
    uint64_t const *vertex_indices;
    // Pointer to the normal index buffer.
    uint64_t const *normal_indices;

    // Number of elements in the position buffer.
    // This should be at least `max(vertex_indices)+1`.
    size_t position_count;
    // Pointer to the position vector buffer.
    // Assumed to be in the following format: XYZ XYZ XYZ
    float const *positions;
    // Pointer to the UV buffer.
    // Assumed to be in the following format: UV UV UV
    float const *uvs;

    // Number of elements in the normal buffer.
    // This should be at least `max(normal_indices)+1`.
    size_t normal_count;
    // Pointer to the normal vector buffer.
    // Assumed to be in the following format: XYZ XYZ XYZ
    float const *normals;
} Trigen_Mesh;

typedef struct Trigen_Transform_t {
    float position[3];
    float orientation[4];
    float scale[3];
} Trigen_Transform;

typedef struct Trigen_Texture_t {
    /** Pointer to the RGB image data; without any padding */
    void const *image;
    /** Image width */
    int width;
    /** Image height */
    int height;
} Trigen_Texture;

typedef enum Trigen_Texture_Kind_t {
    Trigen_Texture_BaseColor = 0,
    Trigen_Texture_NormalMap,
    Trigen_Texture_HeightMap,
    Trigen_Texture_RoughnessMap,
    Trigen_Texture_AmbientOcclusionMap,
    Trigen_Texture_Max,
} Trigen_Texture_Kind;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \brief Creates a new Trigen session.
 * 
 * \param [out] session Pointer to the place where the session handle will be stored;
 * can't be NULL.
 * \param [in] params Pointer to the simulation parameters; can't be NULL.
 */
TRIGEN_RETURN_CODE TRIGEN_API Trigen_CreateSession(
    TRIGEN_HANDLE_ACQUIRE Trigen_Session *session,
    TRIGEN_IN Trigen_Parameters const *params);

/**
 * \brief Destroys a Trigen session, freeing most resources in use.
 *
 * \note
 * Meshes created by \ref Trigen_Mesh_GetMesh won't be freed by a call to this!
 *
 * \param [in] session Session handle
 * 
 */
TRIGEN_RETURN_CODE TRIGEN_API Trigen_DestroySession(
    TRIGEN_HANDLE_RELEASE Trigen_Session session);

/**
 * \brief Creates a collider with a specific mesh and a specific transform.
 *
 * \param [out] collider Pointer to the place where the collider handle will
 * be stored; can't be NULL.
 * \param [in] session Session handler
 * \param [in] mesh Pointer to the mesh descriptor
 * \param [in] transform Pointer to the initial transform of the collider
 */
TRIGEN_RETURN_CODE TRIGEN_API Trigen_CreateCollider(
    TRIGEN_HANDLE_ACQUIRE Trigen_Collider *collider,
    TRIGEN_HANDLE Trigen_Session session,
    TRIGEN_IN Trigen_Collider_Mesh const *mesh,
    TRIGEN_IN Trigen_Transform const *transform);

/**
 * \brief Sets the transform of a specific collider.
 *
 * \param [in] collider Collider handle
 * \param [in] transform New transform
 */
TRIGEN_RETURN_CODE TRIGEN_API Trigen_UpdateCollider(
    TRIGEN_HANDLE Trigen_Collider collider,
    TRIGEN_IN Trigen_Transform const *transform);

/**
 * \brief Grows the plant in the function of the elapsed time.
 *
 * \param [in] session Session handle
 * \param [in] time Elapsed time (in seconds); must be a number but can't be
 * negative
 */
TRIGEN_RETURN_CODE TRIGEN_API Trigen_Grow(
    TRIGEN_HANDLE Trigen_Session session,
    float time);

/**
 * \brief Sets the metaball scale factor.
 *
 * \param [in] session Session handle
 * \param [in] scale Scale factor
 */
TRIGEN_RETURN_CODE TRIGEN_API Trigen_Metaballs_SetScale(
    TRIGEN_HANDLE Trigen_Session session,
    float scale);

/**
 * \brief Regenerates the metaballs.
 * Must be called before calling Trigen_Mesh_Regenerate!
 *
 * \param [in] session Session handle
 */
TRIGEN_RETURN_CODE TRIGEN_API Trigen_Metaballs_Regenerate(
    TRIGEN_HANDLE Trigen_Session session);

/**
 * \brief Sets the number of subdivions to use during mesh generation.
 * The higher the number, the greater the mesh resolution will be.
 * 
 * \param [in] session Session handle
 * \param [in] subdivisions Number of subdivisions, must be at least one.
 */
TRIGEN_RETURN_CODE TRIGEN_API Trigen_Mesh_SetSubdivisions(
    TRIGEN_HANDLE Trigen_Session session,
    int subdivisions);

/**
 * \brief Regenerates the mesh.
 * Must be called at least once before trying to get the mesh using
 * Trigen_Mesh_GetMesh.
 * Must call Trigen_Metaballs_Regenerate before calling this function!
 * 
 * \param [in] session Session handle
 */
TRIGEN_RETURN_CODE TRIGEN_API Trigen_Mesh_Regenerate(
    TRIGEN_HANDLE Trigen_Session session);

/**
 * \brief Makes the generated mesh available to the called code.
 * Pointers to the vertex attributes of the mesh will be put into `mesh`.
 * Must call Trigen_Mesh_Regenerate before calling this function!
 * The contents of the `mesh` structure must be freed using
 * Trigen_Mesh_FreeMesh (destroying the session won't free it)!
 * 
 * \param [in] session Session handle
 * \param [inout] mesh Where to store the mesh data
 */
TRIGEN_RETURN_CODE TRIGEN_API Trigen_Mesh_GetMesh(
    TRIGEN_HANDLE Trigen_Session session,
    TRIGEN_IN Trigen_Mesh *mesh);

/**
 * \brief Frees the memory owned by the `mesh`.
 * 
 * \param [inout] mesh Mesh descriptor
 */
TRIGEN_RETURN_CODE TRIGEN_API Trigen_Mesh_FreeMesh(
    TRIGEN_FREED TRIGEN_INOUT Trigen_Mesh *mesh);

/**
 * \brief Sets one of the input textures.
 * 
 * \param [in] session Session handle
 * \param [in] kind Texture slot identifier
 * \param [in] texture Texture data
 */
TRIGEN_RETURN_CODE TRIGEN_API Trigen_Painting_SetInputTexture(
    TRIGEN_HANDLE Trigen_Session session,
    Trigen_Texture_Kind kind,
    TRIGEN_IN Trigen_Texture const *texture);

/**
 * \brief Sets the resolution of the output textures.
 * 
 * \param [in] session Session handle
 * \param [in] width Texture width
 * \param [in] height Texture height
 */
TRIGEN_RETURN_CODE TRIGEN_API Trigen_Painting_SetOutputResolution(
    TRIGEN_HANDLE Trigen_Session session,
    int width,
    int height);

/**
 * \brief Regenerates the texture paint of the plant.
 * Must be called after Trigen_Mesh_Regenerate and before
 * Trigen_Painting_GetOutputTexture.
 * 
 * \param [in] session Session handle
 */
TRIGEN_RETURN_CODE TRIGEN_API Trigen_Painting_Regenerate(
    TRIGEN_HANDLE Trigen_Session session);

/**
 * \brief Gets one of the output textures.
 * The texture data is owned by the session and doesn't have to be freed.
 * Must call Trigen_Painting_Regenerate first!
 * 
 * \param [in] session Session handle
 * \param [in] kind Texture slot identifier
 * \param [out] texture Where to store the texture info
 */
TRIGEN_RETURN_CODE TRIGEN_API Trigen_Painting_GetOutputTexture(
    TRIGEN_HANDLE Trigen_Session session,
    Trigen_Texture_Kind kind,
    TRIGEN_OUT Trigen_Texture *texture);

/**
 * \brief Translates a status code to a human-readable error message.
 * 
 * \param [out] message Where to put the pointer to the error message
 * \param [in] rc Status code
 */
TRIGEN_RETURN_CODE TRIGEN_API Trigen_GetErrorMessage(
    TRIGEN_OUT char const **message,
    Trigen_Status rc);

/**
 * \brief Makes the structure of the plant available to the caller.
 * The function writes the positions of the endpoints of the branches into the
 * buffer; the data in the buffer will be laid out in the following way:
 * BRANCH_0_POS_0 BRANCH_0_POS_1 BRANCH_1_POS_0 BRANCH_1_POS_1
 * where BRANCH_i_POS_0 is the endpoint of the branch that is connected to it's
 * parent.
 * Each position entry is made of three floats: [x y z].
 *
 * If `buffer` is NULL then the value at `count` will contain the number of
 * branches that would've been written to `buffer`.
 * 
 * If the value at `count` is less than the actual number of branches that the
 * plant has then the function will refuse to write to `buffer` and will return
 * `Trigen_NotEnoughSpace`.
 *
 * \param session Session handle; can't be NULL.
 * \param count Pointer to the number of branches that will fit in the array at
 * `buffer`; can't be NULL.
 * \param buffer Pointer to the array that will store the branches.
 *
 * \returns The status code.
 */
TRIGEN_RETURN_CODE TRIGEN_API Trigen_GetBranches(
    TRIGEN_HANDLE Trigen_Session session,
    TRIGEN_INOUT size_t *count,
    TRIGEN_IN float *buffer);

#ifdef __cplusplus
}
#endif

#endif /* H_EASI_TRIGEN */
