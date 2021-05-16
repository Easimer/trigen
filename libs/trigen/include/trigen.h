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
} Trigen_Status;

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
    float const *position;
    float const *normal;
    size_t const *elements;

    float const *uv;
} Trigen_Mesh;

typedef struct Trigen_Transform_t {
    float position[3];
    float orientation[4];
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

Trigen_Status TRIGEN_API Trigen_CreateSession(Trigen_Session *session, Trigen_Parameters const *params);
Trigen_Status TRIGEN_API Trigen_DestroySession(Trigen_Session session);

Trigen_Status TRIGEN_API Trigen_CreateCollider(Trigen_Collider *collider, Trigen_Session session, Trigen_Collider_Mesh const *mesh, Trigen_Transform const *transform);
Trigen_Status TRIGEN_API Trigen_UpdateCollider(Trigen_Collider collider, Trigen_Transform const *transform);

Trigen_Status TRIGEN_API Trigen_Grow(Trigen_Session session, float time);

Trigen_Status TRIGEN_API Trigen_Metaballs_SetRadius(Trigen_Session session, float radius);
Trigen_Status TRIGEN_API Trigen_Metaballs_Regenerate(Trigen_Session session);

Trigen_Status TRIGEN_API Trigen_Mesh_SetSubdivisions(Trigen_Session session, int subdivisions);
Trigen_Status TRIGEN_API Trigen_Mesh_Regenerate(Trigen_Session session);
Trigen_Status TRIGEN_API Trigen_Mesh_GetMesh(Trigen_Session session, Trigen_Mesh *mesh);

Trigen_Status TRIGEN_API Trigen_Painting_SetInputTexture(Trigen_Session session, Trigen_Texture_Kind kind, Trigen_Texture const *texture);
Trigen_Status TRIGEN_API Trigen_Painting_SetOutputResolution(Trigen_Session session, int width, int height);
Trigen_Status TRIGEN_API Trigen_Painting_Regenerate(Trigen_Session session);
Trigen_Status TRIGEN_API Trigen_Painting_GetOutputTexture(Trigen_Session session, Trigen_Texture_Kind kind, Trigen_Texture *texture);

Trigen_Status TRIGEN_API Trigen_GetErrorMessage(char const **message, Trigen_Status rc);

#ifdef __cplusplus
}
#endif

#endif /* H_EASI_TRIGEN */
