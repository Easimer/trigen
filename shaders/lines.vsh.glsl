#version 330 core
layout (location = 0) in vec3 aPosition;
layout (location = 1) in float fT;

uniform mat4 matModel;
uniform mat4 matView;
uniform mat4 matProj;

out float t;

void main() {
	gl_Position = matProj * matView * matModel * vec4(aPosition.xyz, 1.0);
	t = fT;
}
