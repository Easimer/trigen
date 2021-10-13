#ifndef VAO_LAYOUT
#define VAO_LAYOUT(i) layout (location = i)
#endif

VAO_LAYOUT(0) out vec4 outBaseColor;
VAO_LAYOUT(1) out vec4 outNormal;
VAO_LAYOUT(2) out vec4 outPosition;

in vec3 vPosition;
in vec2 vUV;
in vec3 vNormal;

uniform sampler2D texDiffuse;

void main() {
    outBaseColor = texture(texDiffuse, vUV);
    if (outBaseColor.a < 1) {
        discard;
    }
    outNormal = vec4(vNormal, 1);
    outPosition = vec4(vPosition, 1);
}
