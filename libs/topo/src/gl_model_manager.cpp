// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: 
//

#include <array>
#include <vector>

#include <topo.h>
#include "gl_model_manager.h"

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

namespace topo {

static void TMCMessageCallback(void* user, char const* msg, ETMC_Message_Level level) {
    switch(level) {
        case k_ETMCMsgLevel_Error:
        printf("[ topo ] mesh compressor ERROR: %s\n", msg);
        break;
        case k_ETMCMsgLevel_Info:
        printf("[ topo ] mesh compressor: %s\n", msg);
        break;
    }
}

static void
CalculateTangentsAndBitangents(
    Model_Descriptor const *model,
    std::vector<glm::vec3> &tangents,
    std::vector<glm::vec3> &bitangents) {
    tangents.resize(model->element_count);
    bitangents.resize(model->element_count);

    auto num_triangles = model->element_count / 3;
    for (size_t t = 0; t < num_triangles; t++) {
        auto idx0 = model->elements[t * 3 + 0];
        auto idx1 = model->elements[t * 3 + 1];
        auto idx2 = model->elements[t * 3 + 2];

        auto p0 = ((glm::vec3 *)model->vertices)[idx0];
        auto p1 = ((glm::vec3 *)model->vertices)[idx1];
        auto p2 = ((glm::vec3 *)model->vertices)[idx2];
        auto w0 = ((glm::vec2 *)model->uv)[idx0];
        auto w1 = ((glm::vec2 *)model->uv)[idx1];
        auto w2 = ((glm::vec2 *)model->uv)[idx2];

        auto e1 = p1 - p0;
        auto e2 = p2 - p0;
        auto x1 = w1.x - w0.x;
        auto x2 = w2.x - w0.x;
        auto y1 = w1.y - w0.y;
        auto y2 = w2.y - w0.y;

        auto r = 1.0f / (x1 * y2 - x2 * y1);
        assert(std::isfinite(r) && !std::isnan(r));
        auto tangent = normalize((e1 * y2 - e2 * y1) * r);
        auto bitangent = normalize((e2 * x1 - e1 * x2) * r);

        tangents[t * 3 + 0] = tangent;
        tangents[t * 3 + 1] = tangent;
        tangents[t * 3 + 2] = tangent;
        bitangents[t * 3 + 0] = bitangent;
        bitangents[t * 3 + 1] = bitangent;
        bitangents[t * 3 + 2] = bitangent;
    }
}

static void
CompressModel(
    Model_Descriptor const *model,
    TMC_Context *tmc_context_out,
    TMC_Attribute *attr_position,
    TMC_Attribute *attr_uv,
    TMC_Attribute *attr_tangent,
    TMC_Attribute *attr_bitangent,
    TMC_Attribute *attr_normal) {
    assert(model);
    assert(
        tmc_context_out && attr_position && attr_uv && attr_tangent
        && attr_bitangent && attr_normal);

    *tmc_context_out = nullptr;
    *attr_position = nullptr;
    *attr_uv = nullptr;
    *attr_tangent = nullptr;
    *attr_bitangent = nullptr;
    *attr_normal = nullptr;

    // Vertex positions and normals are already in index-to-direct format
    // We need to turn them into a direct format first
    std::vector<glm::vec3> position;
    std::vector<glm::vec3> normal;
    std::vector<glm::vec2> texcoord;

    auto num_vertices = model->element_count;

    position.resize(num_vertices);
    normal.resize(num_vertices);
    texcoord.resize(num_vertices);
    for (size_t i = 0; i < num_vertices; i++) {
        auto idx = model->elements[i];
        position[i] = ((glm::vec3 *)model->vertices)[idx];
        normal[i] = ((glm::vec3 *)model->normals)[idx];
        texcoord[i] = ((glm::vec2 *)model->uv)[idx];
    }

    TMC_Context ctx = nullptr;
    TMC_Buffer buf_position;
    TMC_Buffer buf_normal;
    TMC_Buffer buf_uv;
    TMC_Buffer buf_tangent;
    TMC_Buffer buf_bitangent;
    TMC_CreateContext(&ctx, k_ETMCHint_AllowSmallerIndices);
    assert(ctx);

    // TMC_SetDebugMessageCallback(ctx, TMCMessageCallback, nullptr);
    TMC_SetParamInteger(ctx, k_ETMCParam_WindowSize, 16);

    TMC_CreateBuffer(
        ctx, &buf_position, position.data(),
        num_vertices * sizeof(position[0]));
    TMC_CreateBuffer(
        ctx, &buf_uv, texcoord.data(), num_vertices * sizeof(texcoord[0]));

    TMC_CreateAttribute(
        ctx, attr_position, buf_position, 3, k_ETMCType_Float32,
        3 * sizeof(float), 0);
    TMC_CreateAttribute(
        ctx, attr_uv, buf_uv, 2, k_ETMCType_Float32, 2 * sizeof(float), 0);

    bool const can_generate_tbn_info
        = model->uv != nullptr && model->normals != nullptr;

    if (can_generate_tbn_info) {
        std::vector<glm::vec3> tangents(num_vertices);
        std::vector<glm::vec3> bitangents(num_vertices);

        CalculateTangentsAndBitangents(model, tangents, bitangents);

        TMC_CreateBuffer(
            ctx, &buf_normal, normal.data(), num_vertices * sizeof(normal[0]));
        TMC_CreateBuffer(
            ctx, &buf_tangent, tangents.data(),
            tangents.size() * sizeof(tangents[0]));
        TMC_CreateBuffer(
            ctx, &buf_bitangent, bitangents.data(),
            bitangents.size() * sizeof(bitangents[0]));

        TMC_CreateAttribute(
            ctx, attr_normal, buf_normal, 3, k_ETMCType_Float32,
            3 * sizeof(float), 0);
        TMC_CreateAttribute(
            ctx, attr_tangent, buf_tangent, 3, k_ETMCType_Float32,
            3 * sizeof(float), 0);
        TMC_CreateAttribute(
            ctx, attr_bitangent, buf_bitangent, 3, k_ETMCType_Float32,
            3 * sizeof(float), 0);
    }

    TMC_Compress(ctx, model->element_count);

    *tmc_context_out = ctx;
}

template<typename T>
static void
CopyAttributeInto(std::unique_ptr<T[]>& arr, TMC_Context ctx, TMC_Attribute attr) {
    void const *data;
    TMC_Size size;
    if (TMC_GetDirectArray(ctx, attr, &data, &size) == k_ETMCStatus_OK) {
        auto count = size / sizeof(T);
        arr = std::make_unique<T[]>(count);
        memcpy(arr.get(), data, size);
    }
}

bool
GL_Model_Manager::CreateModel(
    Model_ID *outHandle,
    Model_Descriptor const *model) {
    if (outHandle == nullptr || model == nullptr) {
        return false;
    }

    if (model->elements == nullptr || model->vertices == nullptr
        || model->uv == nullptr || model->normals == nullptr) {
        return false;
    }

    TMC_Context compress_context;
    TMC_Attribute attr_position, attr_uv, attr_tangent, attr_bitangent, attr_normal;
    // The input mesh is already in index-to-direct format, but we add
    // additional information to the vertices and so we need to recompress
    // it
    CompressModel(
        model, &compress_context, &attr_position, &attr_uv, &attr_tangent,
        &attr_bitangent, &attr_normal);

    Mesh_Data meshData;

    CopyAttributeInto(meshData.positions, compress_context, attr_position);
    CopyAttributeInto(meshData.texcoords, compress_context, attr_uv);
    CopyAttributeInto(meshData.tangents, compress_context, attr_tangent);
    CopyAttributeInto(meshData.bitangents, compress_context, attr_bitangent);
    CopyAttributeInto(meshData.normals, compress_context, attr_normal);

    void const *elements_data = nullptr;
    TMC_Size elements_size = 0;
    TMC_Size elements_count = 0;
    TMC_GetIndexArray(compress_context, &elements_data, &elements_size, &elements_count);

    meshData.elements = std::make_unique<uint8_t[]>(elements_size);
    memcpy(meshData.elements.get(), elements_data, elements_size);

    ETMC_Type index_type;
    TMC_GetIndexArrayType(compress_context, &index_type);

    switch (index_type) {
    case k_ETMCType_UInt16:
        meshData.elementType = GL_UNSIGNED_SHORT;
        break;
    case k_ETMCType_UInt32:
        meshData.elementType = GL_UNSIGNED_INT;
        break;
    }

    meshData.numVertices = model->vertex_count;
    meshData.numElements = elements_count;

    TMC_DestroyContext(compress_context);

    _models.push_back(std::move(meshData));
    *outHandle = &_models.back();

    return false;
}

void
GL_Model_Manager::DestroyModel(Model_ID model) {
    if (model == nullptr)
        return;

    std::remove_if(_models.begin(), _models.end(), [&](Mesh_Data const &m) {
        return &m == model;
    });
}

void
GL_Model_Manager::Regenerate() {
    size_t vec2DataSize = 0;
    size_t vec3DataSize = 0;
    size_t indexDataSize = 0;

    fprintf(stderr, "[ topo ] Regenerating megabuffer...\n");

    for (auto& model : _models) {
        vec2DataSize += model.numVertices * sizeof(glm::vec2);
        vec3DataSize += model.numVertices * sizeof(glm::vec3);

        switch (model.elementType) {
        case GL_UNSIGNED_INT:
            indexDataSize += model.numElements * sizeof(GLuint);
            break;
        case GL_UNSIGNED_SHORT:
            indexDataSize += model.numElements * sizeof(GLushort);
            break;
        }
    }

    fprintf(stderr, "[ topo ] Total vec2 data: %zu bytes\n", vec2DataSize);
    fprintf(stderr, "[ topo ] Total vec3 data: %zu bytes\n", vec3DataSize);
    fprintf(stderr, "[ topo ] Total index data: %zu bytes\n", indexDataSize);

    if (!_buffer) {
        _buffer = Megabuffer();
    } else {
        glDeleteBuffers(1, &_buffer->bufVertices);
        glDeleteBuffers(1, &_buffer->bufElements);
        glDeleteVertexArrays(1, &_buffer->vao);
    }

    glGenVertexArrays(1, &_buffer->vao);
    glBindVertexArray(_buffer->vao);

    glGenBuffers(1, &_buffer->bufVertices);

    auto flags = GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT;
    auto totalVertexDataSize = 4 * vec3DataSize + 1 * vec2DataSize;

    // Generate a buffer for the vertex data
    glBindBuffer(GL_ARRAY_BUFFER, _buffer->bufVertices);
    glBufferStorage(GL_ARRAY_BUFFER, totalVertexDataSize, nullptr, flags);

    auto vertexBuffer = (uint8_t *)glMapBufferRange(GL_ARRAY_BUFFER, 0, totalVertexDataSize, flags);

    // Compute vertex bases
    size_t baseVertex = 0;
    for (auto& model : _models) {
        model.baseVertex = baseVertex;
        baseVertex += model.numVertices;
    }

    size_t vertexBufferWritePtr = 0;

    auto *vertexBufferPositionOffset = (void *)vertexBufferWritePtr;
    // Copy vertex positions
    for (auto &model : _models) {
        auto size = model.numVertices * sizeof(glm::vec3);
        memcpy(&vertexBuffer[vertexBufferWritePtr], model.positions.get(), size);
        vertexBufferWritePtr += size;
    }

    auto *vertexBufferTexcoordOffset = (void *)vertexBufferWritePtr;
    // Copy vertex texcoords
    for (auto &model : _models) {
        auto size = model.numVertices * sizeof(glm::vec2);
        memcpy(&vertexBuffer[vertexBufferWritePtr], model.texcoords.get(), size);
        vertexBufferWritePtr += size;
    }

    auto *vertexBufferNormalOffset = (void *)vertexBufferWritePtr;
    // Copy vertex normals
    for (auto &model : _models) {
        auto size = model.numVertices * sizeof(glm::vec3);
        memcpy(&vertexBuffer[vertexBufferWritePtr], model.normals.get(), size);
        vertexBufferWritePtr += size;
    }

    auto *vertexBufferTangentOffset = (void *)vertexBufferWritePtr;
    // Copy vertex tangents
    for (auto &model : _models) {
        auto size = model.numVertices * sizeof(glm::vec3);
        memcpy(&vertexBuffer[vertexBufferWritePtr], model.tangents.get(), size);
        vertexBufferWritePtr += size;
    }

    auto *vertexBufferBitangentOffset = (void *)vertexBufferWritePtr;
    // Copy vertex bitangents
    for (auto &model : _models) {
        auto size = model.numVertices * sizeof(glm::vec3);
        memcpy(&vertexBuffer[vertexBufferWritePtr], model.bitangents.get(), size);
        vertexBufferWritePtr += size;
    }

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), vertexBufferPositionOffset);
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), vertexBufferTexcoordOffset);
    glEnableVertexAttribArray(1);

    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), vertexBufferNormalOffset);
    glEnableVertexAttribArray(2);

    glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), vertexBufferTangentOffset);
    glEnableVertexAttribArray(3);

    glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), vertexBufferBitangentOffset);
    glEnableVertexAttribArray(4);

    // Create buffer for the indices

    glGenBuffers(1, &_buffer->bufElements);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _buffer->bufElements);
    glBufferStorage(GL_ELEMENT_ARRAY_BUFFER, totalVertexDataSize, nullptr, flags);

    auto indexBuffer = (uint8_t *)glMapBufferRange(GL_ELEMENT_ARRAY_BUFFER, 0, indexDataSize, flags);
    size_t indexBufferWritePtr = 0;

    // Copy indices
    for (auto &model : _models) {
        size_t idxSize;

        switch (model.elementType) {
        case GL_UNSIGNED_INT:
            idxSize = model.numElements * sizeof(GLuint);
            break;
        case GL_UNSIGNED_SHORT:
            idxSize = model.numElements * sizeof(GLushort);
            break;
        default:
            std::abort();
        }

        memcpy(&indexBuffer[indexBufferWritePtr], model.elements.get(), idxSize);

        model.indexOffset = (void *)indexBufferWritePtr;
        indexBufferWritePtr += idxSize;
    }

    glUnmapBuffer(GL_ARRAY_BUFFER);
    glUnmapBuffer(GL_ELEMENT_ARRAY_BUFFER);
}

void
GL_Model_Manager::BindMegabuffer() {
    if (!_buffer)
        return;

    glBindVertexArray(_buffer->vao);
}

void
GL_Model_Manager::GetDrawParameters(
    Model_ID model,
    void **indexOffset,
    GLint *baseVertex,
    GLenum *elementType,
    size_t *numElements) {
    if (model == nullptr || indexOffset == nullptr || baseVertex == nullptr
        || elementType == nullptr || numElements == nullptr) {
        std::abort();
    }

    auto *meshData = ((Mesh_Data *)model);

    *indexOffset = meshData->indexOffset;
    *baseVertex = meshData->baseVertex;
    *elementType = meshData->elementType;
    *numElements = meshData->numElements;
}

}
