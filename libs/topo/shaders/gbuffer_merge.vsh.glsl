// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: deferred rendering lighting pass
//

#ifndef VAO_LAYOUT
#define VAO_LAYOUT(i) layout (location = i)
#endif

VAO_LAYOUT(0) in vec2 aPos;
VAO_LAYOUT(1) in vec2 aUV;

out vec2 vUV;

void main() {
    gl_Position = vec4(aPos, 0, 1);
    vUV = aUV;
}