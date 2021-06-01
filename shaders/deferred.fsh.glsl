// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: deferred rendering lighting pass
//

uniform sampler2D texBaseColor;
uniform sampler2D texNormal;
uniform sampler2D texPosition;

in vec2 vUV;

out vec4 vFrag;

void main() {
    vFrag = vec4(texture(texBaseColor, vUV).rgb, 1);
}