#version 330 core
layout (location = 0) in vec3 aPosition;

uniform mat4 matMVP;

void main() {
	gl_Position = matMVP * vec4(aPosition.xyz, 1.0);
}