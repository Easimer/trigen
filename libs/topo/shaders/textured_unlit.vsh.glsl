#ifndef VAO_LAYOUT
#define VAO_LAYOUT(i) layout(location = i)
#endif

VAO_LAYOUT(0) in vec3 aPosition;
VAO_LAYOUT(1) in vec2 aUV;

layout(std140, binding = 0) uniform Matrices { mat4 matModel[256]; };

uniform mat4 matVP;

invariant gl_Position;

out vec3 vPosition;
out vec2 vUV;

void
main() {
    gl_Position = matVP * matModel[gl_DrawID] * vec4(aPosition.xyz, 1.0);
    vPosition = gl_Position.xyz;
    vUV = vec2(aUV.x, 1 - aUV.y);
}
