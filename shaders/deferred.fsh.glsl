// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: deferred rendering lighting pass
//

uniform sampler2D gPosition;
uniform sampler2D gAlbedo;
uniform sampler2D gSelfIllum;

in vec2 vUV;

out vec4 vFrag;

void main() {
    vFrag = texture(gAlbedo, vUV);
}