#ifndef VAO_LAYOUT
#define VAO_LAYOUT(i) layout (location = i)
#endif

VAO_LAYOUT(0) out vec4 outBaseColor;
VAO_LAYOUT(1) out vec4 outNormal;
VAO_LAYOUT(2) out vec4 outPosition;

uniform vec3 color;

in vec3 vPosition;

void
main() {
    outBaseColor = vec4(color, 1.0f);
    outNormal = vec4(0.0f, 0.0f, 1.0f, 1.0f);
    outPosition = vec4(vPosition, 1.0f);
}
