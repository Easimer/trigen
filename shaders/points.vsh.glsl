#ifndef VAO_LAYOUT
#define VAO_LAYOUT(i) layout (location = i)
#endif

VAO_LAYOUT(0) in vec3 aPosition;

uniform mat4 matModel;
uniform mat4 matView;
uniform mat4 matProj;

void main() {
    gl_Position = matProj * matView * matModel * vec4(aPosition.xyz, 1.0);
}
