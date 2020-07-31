// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: Ellipsoid raymarching vertex shader
//

#ifndef VAO_LAYOUT
#define VAO_LAYOUT(i) layout (location = i)
#endif

VAO_LAYOUT(0) in vec2 aPosition;

out vec2 vUV;

void main() {
	vUV = aPosition;
	gl_Position = vec4(aPosition, 0, 1);
}
