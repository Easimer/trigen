#ifndef VAO_LAYOUT
#define VAO_LAYOUT(i) layout(location = i)
#endif

VAO_LAYOUT(0) in vec3 aPosition;

layout(std140, binding = 0) uniform Matrices {
    mat4 matModel[256];
};

uniform mat4 matVP;

invariant gl_Position;

void
main() {
    gl_Position = matVP * matModel[gl_DrawID] * vec4(aPosition.xyz, 1.0);
}