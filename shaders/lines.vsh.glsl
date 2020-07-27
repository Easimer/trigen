layout (location = 0) in vec3 aPosition;
layout (location = 1) in float fT;

uniform mat4 matMVP;

out float t;

void main() {
	gl_Position = matMVP * vec4(aPosition.xyz, 1.0);
	t = fT;
}
