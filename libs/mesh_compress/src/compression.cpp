// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#include "stdafx.h"

#include "arena.h"
#include "dbgmsg.h"
#include <cstdlib>
#include <cstring>
#include <glm/glm.hpp>
#include <limits>
#include <optional>

#include <Tracy.hpp>

template<typename ArrayType>
static ArrayType
sample_attribute(TMC_HANDLE TMC_Attribute attribute, TMC_Size element) {
    ZoneScoped;
    auto *ptr = attribute->buffer->data.get();
    auto byteOffset = attribute->stride * element + attribute->offset;
    
    if (byteOffset >= attribute->buffer->size) {
        std::abort();
    }

    return *(ArrayType *)&ptr[byteOffset];
}

template<typename ArrayType>
static ArrayType
sample_attribute_arena(
    Arena const &arena,
    TMC_HANDLE TMC_Attribute attribute,
    TMC_Size element) {
    ZoneScoped;
    auto *ptr = static_cast<uint8_t const *>(arena.data());
    auto byteOffset = attribute->stride * element + attribute->offset;
    
    if (byteOffset >= arena.size()) {
        std::abort();
    }

    return *reinterpret_cast<ArrayType const *>(&ptr[byteOffset]);
}

template<typename ArrayType>
static float
calculate_error_between_vertices(
    TMC_HANDLE TMC_Attribute attr,
    TMC_Index origIndex,
    Arena const &arena,
    TMC_Index arenaIndex) {
    ZoneScoped;
    return glm::distance(sample_attribute<ArrayType>(attr, origIndex),
        sample_attribute_arena<ArrayType>(arena, attr, arenaIndex));
}

static float
distance(
    TMC_HANDLE TMC_Context context,
    TMC_Index origIndex,
    std::vector<Arena> const &arenas,
    TMC_Index arenaIndex) {
    ZoneScoped;
    float total_error = 0;

    for (TMC_Index i = 0; i < context->attributes.size(); i++) {
        auto *attr = context->attributes[i].get();
        auto const &arena = arenas[i];

        switch (attr->type) {
        case k_ETMCType_Float32:
            switch (attr->numComponents) {
            case 1:
                total_error += calculate_error_between_vertices<glm::vec<1, float>>(attr, origIndex, arena, arenaIndex);
                break;
            case 2:
                total_error += calculate_error_between_vertices<glm::vec<2, float>>(attr, origIndex, arena, arenaIndex);
                break;
            case 3:
                total_error += calculate_error_between_vertices<glm::vec<3, float>>(attr, origIndex, arena, arenaIndex);
                break;
            case 4:
                total_error += calculate_error_between_vertices<glm::vec<4, float>>(attr, origIndex, arena, arenaIndex);
                break;
            default:
                std::abort();
                break;
            }
            break;
        default:
            assert(!"Unhandled attribute type");
            break;
        }
    }

    return total_error;
}

template<typename ArrayType>
static void
copy(
    Arena &arena,
    TMC_HANDLE TMC_Attribute attr,
    TMC_Index idx) {
    ZoneScoped;
    *arena.allocate<ArrayType>() = sample_attribute<ArrayType>(attr, idx);
}

static void
push(
    std::vector<Arena> &arenas,
    TMC_HANDLE TMC_Context ctx,
    TMC_Index idx) {
    ZoneScoped;
    assert(ctx);
    assert(arenas.size() == ctx->attributes.size());
    assert(idx >= 0);

    for (TMC_Index i = 0; i < ctx->attributes.size(); i++) {
        auto *attr = ctx->attributes[i].get();
        auto &arena = arenas[i];

        switch (attr->type) {
        case k_ETMCType_Float32:
            switch (attr->numComponents) {
            case 1:
                copy<glm::vec<1, float>>(arena, attr, idx);
                break;
            case 2:
                copy<glm::vec<2, float>>(arena, attr, idx);
                break;
            case 3:
                copy<glm::vec<3, float>>(arena, attr, idx);
                break;
            case 4:
                copy<glm::vec<4, float>>(arena, attr, idx);
                break;
            default:
                std::abort();
                break;
            }
            break;
        default:
            assert(!"Unhandled attribute type");
            break;
        }
    }
}

template<typename From, typename To>
static void
convert_index_buffer(
    TMC_HANDLE TMC_Context context) {
    ZoneScoped;
    auto elementCount = context->indexBufferCount;

    auto oldIndexBuffer = std::move(context->indexBuffer);
    auto const *oldIndexBufferPtr = (From const *)oldIndexBuffer.get();

    auto newIndexBuffer
        = std::make_unique<uint8_t[]>(elementCount * sizeof(To));
    auto *newIndexBufferPtr = (To *)newIndexBuffer.get();

    for (TMC_Index idx = 0; idx < elementCount; idx++) {
        assert(oldIndexBufferPtr[idx] <= std::numeric_limits<To>::max());
        newIndexBufferPtr[idx] = To(oldIndexBufferPtr[idx]);
    }

    context->indexBuffer = std::move(newIndexBuffer);
    context->indexBufferSize = elementCount * sizeof(To);
}

static void
try_fit_indices_into_smaller_type(
    TMC_HANDLE TMC_Context context) {
    ZoneScoped;
    assert(context);
    assert(context->hints & k_ETMCHint_AllowSmallerIndices);

    switch (context->indexType) {
    case k_ETMCType_UInt32:
        if (context->numOutputVertices < 65536) {
            context->indexType = k_ETMCType_UInt16;
            convert_index_buffer<uint32_t, uint16_t>(context);
        }
        break;
    default:
        std::abort();
        break;
    }
}

template<typename IndexType>
static TMC_RETURN_CODE
compress(
    TMC_HANDLE TMC_Context context,
    TMC_Index vertexCount) {
    ZoneScoped;
    assert(context);
    assert(vertexCount >= 0);
    float const epsilon = 0.01f;

    // TODO(danielm): check whether we have enough data in the buffers for each
    // attribute

    std::vector<IndexType> indexBuffer;
    TMC_Index num_output_vertices = 0;
    TMC_Index last_index = 0;
    std::vector<Arena> arenas(context->attributes.size());

    indexBuffer.reserve(vertexCount);

    for (TMC_Index i = 0; i < vertexCount; i++) {
        std::optional<TMC_Index> idx;

        for (TMC_Index j = num_output_vertices - 1; j >= last_index; j--) {
            if (distance(context, i, arenas, j) < epsilon) {
                idx = j;
                break;
            }
        }

        if (!idx.has_value()) {
            idx = num_output_vertices;
            push(arenas, context, i);
            num_output_vertices++;

            if (context->windowSize != 0) {
                last_index = num_output_vertices - 1 - context->windowSize;
                if (last_index < 0) {
                    last_index = 0;
                }
            }
        }

        assert(idx.has_value());
        indexBuffer.push_back(idx.value());
    }

    assert(indexBuffer.size() == vertexCount);

    for (TMC_Index i = 0; i < context->attributes.size(); i++) {
        context->attributes[i]->compressedBuf = std::move(arenas[i]);
    }

    auto indexBufferBuf = std::make_unique<uint8_t[]>(indexBuffer.size() * sizeof(IndexType));
    memcpy(indexBufferBuf.get(), indexBuffer.data(), indexBuffer.size() * sizeof(IndexType));
    context->indexBuffer = std::move(indexBufferBuf);
    context->indexBufferSize = indexBuffer.size() * sizeof(IndexType);
    context->indexBufferCount = indexBuffer.size();

    context->numOutputVertices = num_output_vertices;

    if (context->hints & k_ETMCHint_AllowSmallerIndices) {
        try_fit_indices_into_smaller_type(context);
    }

    TMC_Print(context, "Compressed mesh");
    TMC_Print(context, "- Input vertex count: %zu", vertexCount);
    TMC_Print(context, "- Output vertex count: %zd", num_output_vertices);
    TMC_Print(context, "  - Index buffer size: %zd B", TMC_Size(context->indexBufferSize));
    TMC_Print(context, "- Attributes:");

    for (TMC_Index i = 0; i < context->attributes.size(); i++) {
        auto &attr = context->attributes[i];
        TMC_Print(context, "  - [#%zd]", i);
        switch (attr->type) {
        case k_ETMCType_Float32:
            TMC_Print(context, "    - Type: %u x float32", attr->numComponents);
            break;
        default:
            assert(0);
            break;
        }
        auto size_bytes = attr->compressedBuf.size();
        auto size_kilobytes = TMC_Size(size_bytes / 1024);
        auto size_megabytes = TMC_Size(size_kilobytes / 1024);
        TMC_Print(context, "    - Compressed size: %zu B | %zu KiB | %zu MiB",
            size_bytes, size_kilobytes, size_megabytes);
    }

    return k_ETMCStatus_OK;
}

HEDLEY_BEGIN_C_DECLS

TMC_API
TMC_RETURN_CODE
TMC_Compress(
    TMC_HANDLE TMC_Context context,
    TMC_Size vertex_count) {
    ZoneScoped;
    if (context == nullptr) {
        return k_ETMCStatus_InvalidArguments;
    }

    switch (context->indexType) {
    case k_ETMCType_UInt16:
        compress<uint16_t>(context, vertex_count);
        break;
    case k_ETMCType_UInt32:
        compress<uint32_t>(context, vertex_count);
        break;
    default:
        std::abort();
        break;
    }

    return k_ETMCStatus_OK;
}

HEDLEY_END_C_DECLS
