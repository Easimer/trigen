// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#include "trigen.h"
#include "trigen.hpp"

extern "C" {

Trigen_Status TRIGEN_API Trigen_CreateSession(Trigen_Session *session, Trigen_Parameters const *params) {
    *session = nullptr;
    return Trigen_OK;
}

Trigen_Status TRIGEN_API Trigen_DestroySession(Trigen_Session session) {
    return Trigen_OK;
}

Trigen_Status TRIGEN_API Trigen_CreateCollider(Trigen_Collider *collider, Trigen_Session session, Trigen_Collider_Mesh const *mesh, Trigen_Transform const *transform) {
    return Trigen_OK;
}

Trigen_Status TRIGEN_API Trigen_UpdateCollider(Trigen_Collider collider, Trigen_Transform const *transform) {
    return Trigen_OK;
}

Trigen_Status TRIGEN_API Trigen_Grow(Trigen_Session session, float time) {
    return Trigen_OK;
}

Trigen_Status TRIGEN_API Trigen_Mesh_SetSubdivisions(Trigen_Session session, int subdivisions) {
    return Trigen_OK;
}

Trigen_Status TRIGEN_API Trigen_Mesh_Regenerate(Trigen_Session session) {
    return Trigen_OK;
}

Trigen_Status TRIGEN_API Trigen_Mesh_GetMesh(Trigen_Session session, Trigen_Mesh const *mesh) {
    return Trigen_OK;
}

Trigen_Status TRIGEN_API Trigen_Painting_SetInputTexture(Trigen_Session session, Trigen_Texture_Kind kind, Trigen_Texture const *texture) {
    return Trigen_OK;
}

Trigen_Status TRIGEN_API Trigen_Painting_SetOutputResolution(Trigen_Session session, int width, int height) {
    return Trigen_OK;
}

Trigen_Status TRIGEN_API Trigen_Painting_Regenerate(Trigen_Session session) {
    return Trigen_OK;
}

Trigen_Status TRIGEN_API Trigen_Painting_GetOutputTexture(Trigen_Session session, Trigen_Texture_Kind kind, Trigen_Texture const **texture) {
    return Trigen_OK;
}

}
