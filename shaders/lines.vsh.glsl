#ifndef VAO_LAYOUT
#define VAO_LAYOUT(i) layout (location = i)
#endif

VAO_LAYOUT(0) in vec3 aPosition;
VAO_LAYOUT(1) in float fT;

uniform mat4 matMVP;

out float t;
out vec3 inPosition;
out vec4 inNormal;

void main() {
    gl_Position = matMVP * vec4(aPosition.xyz, 1.0);
    t = fT;
    inPosition = aPosition.xyz;
    inNormal = matMVP * vec4(0, 0, 1, 1);
}
