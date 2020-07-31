#ifndef VAO_LAYOUT
#define VAO_LAYOUT(i) layout (location = i)
#endif

VAO_LAYOUT(0) in vec3 aPosition;

uniform mat4 matMVP;

void main() {
	gl_Position = matMVP * vec4(aPosition.xyz, 1.0);
}