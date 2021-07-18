#extension GL_ARB_compute_variable_group_size : require

layout(local_size_variable) in;

layout(std140, binding = BINDING_TRANSLATE) buffer TranslateBuffer {
    mat4 Translate[];
};

layout(std140, binding = BINDING_ROTATE) buffer RotateBuffer {
    mat4 Rotate[];
};

layout(std140, binding = BINDING_SCALE) buffer ScaleBuffer {
    mat4 Scale[];
};

layout(std140, binding = BINDING_OUTPUT) buffer OutputBuffer {
    mat4 Output[];
};

uniform uint uiTotalItemCount;

void main() {
    uint id = gl_GlobalInvocationID.x;

    if(id >= uiTotalItemCount) {
        return;
    }

    Output[id] = Translate[id] * Rotate[id] * Scale[id];
}