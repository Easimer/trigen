// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: Ellipsoid raymarching vertex shader
//

#ifndef VAO_LAYOUT
#define VAO_LAYOUT(i) layout (location = i)
#endif

VAO_LAYOUT(0) in vec2 aPosition;

out vec4 vUV;

void main() {
    gl_Position = vec4(aPosition, 0, 1);
    vUV = gl_Position;
}
