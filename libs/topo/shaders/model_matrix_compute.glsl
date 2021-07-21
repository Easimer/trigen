#extension GL_ARB_compute_variable_group_size : require

layout(local_size_variable) in;

layout(std140, binding = BINDING_TRANSLATE) readonly buffer TranslateBuffer {
    vec3 Translate[];
};

layout(std140, binding = BINDING_ROTATE) readonly buffer RotateBuffer {
    mat4 Rotate[];
};

layout(std140, binding = BINDING_SCALE) readonly buffer ScaleBuffer {
    vec3 Scale[];
};

layout(std140, binding = BINDING_OUTPUT) writeonly buffer OutputBuffer {
    mat4 Output[];
};

uniform uint uiTotalItemCount;

void main() {
    uint id = gl_GlobalInvocationID.x;

    if(id >= uiTotalItemCount) {
        return;
    }

    mat4 matTranslate = mat4(1.0);
    matTranslate[3] = vec4(Translate[id], 1);

    mat4 matScale = mat4(1.0);
    vec3 vecScale = Scale[id];
    matScale[0].x = vecScale.x;
    matScale[1].y = vecScale.y;
    matScale[2].z = vecScale.z;

    Output[id] = matTranslate * Rotate[id] * matScale;
}