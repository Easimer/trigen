// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: 
//

#pragma once

#include <list>
#include <optional>

#include <topo.h>

#include <trigen/mesh_compress.h>

#include <glad/glad.h>

namespace topo {

struct Mesh_Data {
    size_t numVertices;
    size_t numElements;

    std::unique_ptr<float[]> positions;
    std::unique_ptr<float[]> texcoords;
    std::unique_ptr<float[]> normals;
    std::unique_ptr<float[]> tangents;
    std::unique_ptr<float[]> bitangents;

    std::unique_ptr<uint8_t[]> elements;
    GLenum elementType;

    void *indexOffset;
    GLint baseVertex;
};

struct Megabuffer {
    GLuint vao;
    GLuint bufVertices, bufElements;
};

class GL_Model_Manager {
public:
    bool
    CreateModel(Model_ID *outHandle, Model_Descriptor const *descriptor);

    void
    DestroyModel(Model_ID model);

    void
    Regenerate();

    void
    BindMegabuffer();

    void
    GetDrawParameters(
        Model_ID model,
        void **indexOffset,
        GLint *baseVertex,
        GLenum *elementType,
        size_t *numElements);

private:
    std::list<Mesh_Data> _models;
    std::optional<Megabuffer> _buffer;
};
}
