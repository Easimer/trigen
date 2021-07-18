#ifndef VAO_LAYOUT
#define VAO_LAYOUT(i) layout (location = i)
#endif

VAO_LAYOUT(0) out vec4 outBaseColor;
VAO_LAYOUT(1) out vec4 outNormal;
VAO_LAYOUT(2) out vec4 outPosition;

in vec3 vPosition;
in vec2 vUV;
in mat3 vTBN;

uniform sampler2D texDiffuse;
uniform sampler2D texNormal;

void main() {
    vec3 normal = texture(texNormal, vUV).rgb;
    normal = normalize(normal * 2.0 - 1.0);
    normal = vTBN * normal;

    outBaseColor = texture(texDiffuse, vUV);
    outNormal = vec4(normal, 1);
    outPosition = vec4(vPosition, 1);
}