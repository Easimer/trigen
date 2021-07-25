#ifndef VAO_LAYOUT
#define VAO_LAYOUT(i) layout(location = i)
#endif

#if __VERSION__ < 460
#extension GL_ARB_shader_draw_parameters : require
#define gl_DrawID gl_DrawIDARB
#endif

VAO_LAYOUT(0) in vec3 aPosition;
VAO_LAYOUT(1) in vec2 aUV;
VAO_LAYOUT(2) in vec3 aNormal;

layout(std140, binding = 0) readonly buffer Matrices {
    mat4 matModels[];
};

layout(std430, binding = 1) readonly buffer MatrixIndices {
    uint idxMatModels[];
};

uniform mat4 matVP;

invariant gl_Position;

out vec3 vPosition;
out vec2 vUV;
out vec3 vNormal;

void
main() {
    uint idxMatModel = idxMatModels[gl_DrawID];
    vec4 worldPosition = matModels[idxMatModel] * vec4(aPosition.xyz, 1.0);
    gl_Position = matVP * worldPosition;
    vPosition = worldPosition.xyz;
    vUV = vec2(aUV.x, 1 - aUV.y);
    mat3 matModelRot = mat3(matModels[idxMatModel]);
    vNormal = normalize(matModelRot * aNormal);
}
