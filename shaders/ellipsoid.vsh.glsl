#version 330 core
layout (location = 0) in vec2 aPosition;

out vec4 vUV;

void main() {
	vUV = vec4(aPosition, 0, 0);
	gl_Position = vec4(aPosition, 0, 1);
}
