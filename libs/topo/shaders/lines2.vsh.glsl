#ifndef VAO_LAYOUT
#define VAO_LAYOUT(i) layout (location = i)
#endif

VAO_LAYOUT(0) in vec3 aPosition;

uniform mat4 matMVP;

out vec3 vPosition;

void
main() {
    gl_Position = matMVP * vec4(aPosition.xyz, 1.0);
    vPosition = gl_Position.xyz;
}
