#ifndef VAO_LAYOUT
#define VAO_LAYOUT(i) layout (location = i)
#endif

VAO_LAYOUT(0) in vec3 aPosition;
VAO_LAYOUT(1) in vec2 aUV;
VAO_LAYOUT(2) in vec3 aNormal;
VAO_LAYOUT(3) in vec3 aTangent;
VAO_LAYOUT(4) in vec3 aBitangent;

uniform mat4 matMVP;
uniform mat4 matModel;

out vec3 vPosition;
out vec3 vNormal;

invariant gl_Position;

void
main() {
    gl_Position = matMVP * vec4(aPosition.xyz, 1.0);
    vPosition = gl_Position.xyz;
    mat3 matModelRot = mat3(matModel);
    vNormal = normalize(matModelRot * aNormal);
}
