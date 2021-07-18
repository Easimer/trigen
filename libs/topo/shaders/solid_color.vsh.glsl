#ifndef VAO_LAYOUT
#define VAO_LAYOUT(i) layout (location = i)
#endif

VAO_LAYOUT(0) in vec3 aPosition;
VAO_LAYOUT(1) in vec2 aUV;
VAO_LAYOUT(2) in vec3 aNormal;
VAO_LAYOUT(3) in vec3 aTangent;
VAO_LAYOUT(4) in vec3 aBitangent;

layout(std140, binding = 0) buffer Matrices { mat4 matModel[]; };
uniform mat4 matVP;

out vec3 vPosition;
out vec3 vNormal;

invariant gl_Position;

void
main() {
    vec4 worldPosition = matModel[gl_DrawID] * vec4(aPosition.xyz, 1.0);
    gl_Position = matVP * worldPosition;
    vPosition = worldPosition.xyz;
    mat3 matModelRot = mat3(matModel[gl_DrawID]);
    vNormal = normalize(matModelRot * aNormal);
}
