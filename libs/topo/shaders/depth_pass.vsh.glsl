#ifndef VAO_LAYOUT
#define VAO_LAYOUT(i) layout(location = i)
#endif

VAO_LAYOUT(0) in vec3 aPosition;

layout(std140, binding = 0) readonly buffer Matrices {
    mat4 matModels[];
};

layout(std430, binding = 1) readonly buffer MatrixIndices {
    uint idxMatModels[];
};

uniform mat4 matVP;

invariant gl_Position;

void
main() {
    uint idxMatModel = idxMatModels[gl_DrawID];
    gl_Position = matVP * matModels[idxMatModel] * vec4(aPosition.xyz, 1.0);
}