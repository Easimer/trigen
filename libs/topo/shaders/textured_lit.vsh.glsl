#ifndef VAO_LAYOUT
#define VAO_LAYOUT(i) layout (location = i)
#endif

VAO_LAYOUT(0) in vec3 aPosition;
VAO_LAYOUT(1) in vec2 aUV;
VAO_LAYOUT(2) in vec3 aNormal;
VAO_LAYOUT(3) in vec3 aTangent;
VAO_LAYOUT(4) in vec3 aBitangent;

layout(std140, binding = 0) uniform Matrices { mat4 matModel[256]; };
uniform mat4 matVP;

invariant gl_Position;

out vec3 vPosition;
out vec2 vUV;

out mat3 vTBN;

void
main() {
    vec4 worldPosition = matModel[gl_DrawID] * vec4(aPosition.xyz, 1.0);
    gl_Position = matVP * worldPosition;
    vPosition = worldPosition.xyz;
    vUV = vec2(aUV.x, 1 - aUV.y);
    mat3 matModelRot = mat3(matModel[gl_DrawID]);
    vec3 T = normalize(matModelRot * aTangent);
    vec3 B = normalize(matModelRot * aBitangent);
    vec3 N = normalize(matModelRot * aNormal);
    vTBN = mat3(T, B, N);
}
