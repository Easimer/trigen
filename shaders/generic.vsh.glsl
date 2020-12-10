#ifndef VAO_LAYOUT
#define VAO_LAYOUT(i) layout (location = i)
#endif

VAO_LAYOUT(0) in vec3 aPosition;

#ifdef GENERIC_SHADER_WITH_VERTEX_COLORS
VAO_LAYOUT(1) in vec3 aColor;
#endif

uniform mat4 matMVP;

#ifdef GENERIC_SHADER_WITH_VERTEX_COLORS
out vec3 vColor;
#endif

void main() {
#ifdef GENERIC_SHADER_WITH_VERTEX_COLORS
    vColor = vec3(aColor.r / 255.0f, aColor.g / 255.0f, aColor.b / 255.0f);
#endif

    gl_Position = matMVP * vec4(aPosition.xyz, 1.0);
}