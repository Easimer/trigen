/* Ideally this would be a compile definition. */
#define TMC_TARGET_VERSION TMC_VERSION_1_0
#include <trigen/mesh_compress.h>

#include <assert.h>
#include <inttypes.h>
#include <stdlib.h>
#include <stdio.h>

#define TMC_CHECK_STATUS(status)                                               \
    if ((status) != k_ETMCStatus_OK)                                          \
    abort();

/**
 * Compress a quad (two triangles)
 * Each vertex has a 3D position and a UV texcoord.
 * Each attribute is in a separate buffer.
 */
static void
quad_separate_buffers() {
    TMC_Context ctx;
    ETMC_Status status;
    TMC_Buffer bufPosition, bufTexcoord;
    TMC_Attribute attrPosition, attrTexcoord;

    float position[] = {
        -1, +1, 0,
        -1, -1, 0,
        +1, +1, 0,

        +1, +1, 0,
        -1, -1, 0,
        +1, -1, 0,
    };

    float texcoord[] = {
        0, 1,
        0, 0,
        1, 1,

        1, 1,
        0, 0,
        1, 0,
    };

    printf("=====\nQuad (separate buffers)\n");

    /* Create a compressor context */
    status = TMC_CreateContext(&ctx, k_ETMCHint_None);
    TMC_CHECK_STATUS(status);

    /* Load the vertex data into the compressor */
    status = TMC_CreateBuffer(ctx, &bufPosition, position, sizeof(position));
    TMC_CHECK_STATUS(status);

    status = TMC_CreateBuffer(ctx, &bufTexcoord, texcoord, sizeof(texcoord));
    TMC_CHECK_STATUS(status);

    /* Create attributes for each vertex attribute */
    status = TMC_CreateAttribute(
        ctx, &attrPosition, bufPosition, 3, k_ETMCType_Float32,
        3 * sizeof(float), 0);
    TMC_CHECK_STATUS(status);

    status = TMC_CreateAttribute(
        ctx, &attrTexcoord, bufTexcoord, 2, k_ETMCType_Float32,
        2 * sizeof(float), 0);
    TMC_CHECK_STATUS(status);

    /* Set the index type to unsigned 32-bit integer */
    status = TMC_SetIndexArrayType(ctx, k_ETMCType_UInt32);
    TMC_CHECK_STATUS(status);

    /* Perform the compression */
    status = TMC_Compress(ctx, 6);
    TMC_CHECK_STATUS(status);

    /* Get the pointers to the results */
    const void *indexBuffer;
    TMC_Size indexBufferCount;
    const void *positionBuffer;
    TMC_Size positionBufferSize;
    const void *texcoordBuffer;
    TMC_Size texcoordBufferSize;

    status = TMC_GetIndexArray(ctx, &indexBuffer, NULL, &indexBufferCount);
    TMC_CHECK_STATUS(status);
    assert(indexBufferCount == 6);

    status = TMC_GetDirectArray(
        ctx, attrPosition, &positionBuffer, &positionBufferSize);
    TMC_CHECK_STATUS(status);

    status = TMC_GetDirectArray(
        ctx, attrTexcoord, &texcoordBuffer, &texcoordBufferSize);
    TMC_CHECK_STATUS(status);

    printf("Index buffer element count: %" PRIu32 "\n", indexBufferCount);
    printf("Position buffer size before: %d after: %" PRIu32 "\n", (int)sizeof(position), positionBufferSize);
    printf("Texcoord buffer size before: %d after: %" PRIu32 "\n", (int)sizeof(texcoord), texcoordBufferSize);
    for (TMC_Size i = 0; i < indexBufferCount; i++) {
        uint32_t idx = ((uint32_t *)indexBuffer)[i];
        const float *pos = ((const float *)positionBuffer) + idx * 3;
        const float *uv = ((const float *)texcoordBuffer) + idx * 2;
        printf(
            "#%" PRIu32 " -> #%" PRIu32 "[ (%f, %f, %f), (%f, %f) ]\n", i, idx,
            pos[0], pos[1], pos[2], uv[0], uv[1]);
    }

    /* Free the context */
    status = TMC_DestroyContext(ctx);
    TMC_CHECK_STATUS(status);
}

/**
 * Compress a quad (two triangles)
 * Each vertex has a 3D position and a UV texcoord.
 * The attributes are interleaved into a single contiguous array.
 */
static void
quad_interleaved_attributes() {
    TMC_Context ctx;
    ETMC_Status status;
    TMC_Buffer bufVertices;
    TMC_Attribute attrPosition, attrTexcoord;

    float vertices[] = {
        -1, +1, 0, /**/ 0, 1,
        -1, -1, 0, /**/ 0, 0,
        +1, +1, 0, /**/ 1, 1,

        +1, +1, 0, /**/ 1, 1,
        -1, -1, 0, /**/ 0, 0,
        +1, -1, 0, /**/ 1, 0,
    };

    printf("=====\nQuad (interleaved)\n");

    /* Create a compressor context */
    status = TMC_CreateContext(&ctx, k_ETMCHint_None);
    TMC_CHECK_STATUS(status);

    /* Load the vertex data into the compressor */
    status = TMC_CreateBuffer(ctx, &bufVertices, vertices, sizeof(vertices));
    TMC_CHECK_STATUS(status);

    /* Create attributes for each vertex attribute
     * Note the values of the stride and offset parameter
     */
    status = TMC_CreateAttribute(
        ctx, &attrPosition, bufVertices, 3, k_ETMCType_Float32,
        5 * sizeof(float), 0);
    TMC_CHECK_STATUS(status);

    status = TMC_CreateAttribute(
        ctx, &attrTexcoord, bufVertices, 2, k_ETMCType_Float32,
        5 * sizeof(float), 3 * sizeof(float));
    TMC_CHECK_STATUS(status);

    /* Set the index type to unsigned 32-bit integer */
    status = TMC_SetIndexArrayType(ctx, k_ETMCType_UInt32);
    TMC_CHECK_STATUS(status);

    /* Perform the compression */
    status = TMC_Compress(ctx, 6);
    TMC_CHECK_STATUS(status);

    /* Get the pointers to the results */
    const void *indexBuffer;
    TMC_Size indexBufferCount;
    const void *positionBuffer;
    TMC_Size positionBufferSize;
    const void *texcoordBuffer;
    TMC_Size texcoordBufferSize;

    status = TMC_GetIndexArray(ctx, &indexBuffer, NULL, &indexBufferCount);
    TMC_CHECK_STATUS(status);
    assert(indexBufferCount == 6);

    status = TMC_GetDirectArray(
        ctx, attrPosition, &positionBuffer, &positionBufferSize);
    TMC_CHECK_STATUS(status);

    status = TMC_GetDirectArray(
        ctx, attrTexcoord, &texcoordBuffer, &texcoordBufferSize);
    TMC_CHECK_STATUS(status);

    printf("Index buffer element count: %" PRIu32 "\n", indexBufferCount);
    printf("Vertex data total size before: %d after: %" PRIu64 "\n", (int)sizeof(vertices), ((uint64_t)positionBufferSize) + texcoordBufferSize);
    for (TMC_Size i = 0; i < indexBufferCount; i++) {
        uint32_t idx = ((uint32_t *)indexBuffer)[i];
        const float *pos = ((const float *)positionBuffer) + idx * 3;
        const float *uv = ((const float *)texcoordBuffer) + idx * 2;
        printf(
            "#%" PRIu32 " -> #%" PRIu32 "[ (%f, %f, %f), (%f, %f) ]\n", i, idx,
            pos[0], pos[1], pos[2], uv[0], uv[1]);
    }

    /* Free the context */
    status = TMC_DestroyContext(ctx);
    TMC_CHECK_STATUS(status);
}

int
main(int argc, char **argv) {
    quad_separate_buffers();
    quad_interleaved_attributes();
    return 0;
}
