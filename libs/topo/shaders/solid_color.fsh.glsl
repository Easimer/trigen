#ifndef VAO_LAYOUT
#define VAO_LAYOUT(i) layout (location = i)
#endif

VAO_LAYOUT(0) out vec4 outBaseColor;
VAO_LAYOUT(1) out vec4 outNormal;
VAO_LAYOUT(2) out vec4 outPosition;

uniform vec3 solidColor;

in vec3 vPosition;
in vec3 vNormal;

void
main() {
    outBaseColor = vec4(solidColor, 1.0f);
    outNormal = vec4(vNormal, 1.0f);
    outPosition = vec4(vPosition, 1.0f);
}
