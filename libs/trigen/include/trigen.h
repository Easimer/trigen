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

#if _WIN32 && defined(_In_) && defined(_Out_) && defined(_Inout_)
#define TRIGEN_IN _In_
#define TRIGEN_OUT _Out_
#define TRIGEN_INOUT _Inout_
#else
#define TRIGEN_IN
#define TRIGEN_OUT
#define TRIGEN_INOUT
#endif

typedef struct Trigen_Session_t* Trigen_Session;
typedef struct Trigen_Collider_t *Trigen_Collider;

typedef enum Trigen_Status_t {
    Trigen_OK = 0,
    Trigen_Failure,
    Trigen_InvalidArguments,
    Trigen_OutOfMemory,
    Trigen_InvalidConfiguration,
    Trigen_InvalidMesh,
    Trigen_NotReady,
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
    Trigen_F_None       = 0,
    Trigen_F_PreferCPU  = 1 << 0,
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
    void const *image;
    int width, height;
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

TRIGEN_RETURN_CODE TRIGEN_API Trigen_CreateSession(
    TRIGEN_OUT Trigen_Session *session,
    TRIGEN_IN Trigen_Parameters const *params);

TRIGEN_RETURN_CODE TRIGEN_API Trigen_DestroySession(TRIGEN_IN Trigen_Session session);

TRIGEN_RETURN_CODE TRIGEN_API Trigen_CreateCollider(
    TRIGEN_OUT Trigen_Collider *collider,
    TRIGEN_IN Trigen_Session session,
    TRIGEN_IN Trigen_Collider_Mesh const *mesh,
    TRIGEN_IN Trigen_Transform const *transform);

TRIGEN_RETURN_CODE TRIGEN_API Trigen_UpdateCollider(Trigen_Collider collider, Trigen_Transform const *transform);

TRIGEN_RETURN_CODE TRIGEN_API Trigen_Grow(Trigen_Session session, float time);

TRIGEN_RETURN_CODE TRIGEN_API Trigen_Metaballs_SetRadius(Trigen_Session session, float radius);
TRIGEN_RETURN_CODE TRIGEN_API Trigen_Metaballs_Regenerate(Trigen_Session session);

TRIGEN_RETURN_CODE TRIGEN_API Trigen_Mesh_SetSubdivisions(Trigen_Session session, int subdivisions);
TRIGEN_RETURN_CODE TRIGEN_API Trigen_Mesh_Regenerate(Trigen_Session session);
TRIGEN_RETURN_CODE TRIGEN_API Trigen_Mesh_GetMesh(Trigen_Session session, Trigen_Mesh *mesh);
TRIGEN_RETURN_CODE TRIGEN_API Trigen_Mesh_FreeMesh(Trigen_Mesh *mesh);

TRIGEN_RETURN_CODE TRIGEN_API Trigen_Painting_SetInputTexture(Trigen_Session session, Trigen_Texture_Kind kind, Trigen_Texture const *texture);
TRIGEN_RETURN_CODE TRIGEN_API Trigen_Painting_SetOutputResolution(Trigen_Session session, int width, int height);
TRIGEN_RETURN_CODE TRIGEN_API Trigen_Painting_Regenerate(Trigen_Session session);
TRIGEN_RETURN_CODE TRIGEN_API Trigen_Painting_GetOutputTexture(Trigen_Session session, Trigen_Texture_Kind kind, Trigen_Texture *texture);

TRIGEN_RETURN_CODE TRIGEN_API Trigen_GetErrorMessage(char const **message, Trigen_Status rc);

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
TRIGEN_RETURN_CODE TRIGEN_API Trigen_GetBranches(TRIGEN_IN Trigen_Session session, TRIGEN_INOUT size_t *count, TRIGEN_IN float *buffer);

#ifdef __cplusplus
}
#endif

#endif /* H_EASI_TRIGEN */
