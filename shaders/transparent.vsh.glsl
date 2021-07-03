#ifndef VAO_LAYOUT
#define VAO_LAYOUT(i) layout (location = i)
#endif

VAO_LAYOUT(0) in vec3 aPosition;

VAO_LAYOUT(1) in vec2 aUV;
out vec2 vUV;

uniform mat4 matMVP;

void main() {
    gl_Position = matMVP * vec4(aPosition.xyz, 1.0);
    vUV = vec2(aUV.x, 1 - aUV.y);
}
