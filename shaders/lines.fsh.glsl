#ifndef VAO_LAYOUT
#define VAO_LAYOUT(i) layout (location = i)
#endif

in float t;
in vec3 inPosition;

// VAO_LAYOUT(0) out vec4 vPosition;
// VAO_LAYOUT(1) out vec4 vAlbedo;
out vec4 vAlbedo;
// VAO_LAYOUT(2) out vec4 vSelfIllum;

uniform vec3 vColor0, vColor1;

void main() {
    vec3 col0 = vColor0;
    vec3 col1 = vColor1;
    vec3 color = (1 - t) * col0 + t * col1;
    // vPosition = vec4(inPosition, 1.0);
    vAlbedo = vec4(color, 1.0);
    // vSelfIllum = vec4(color, 1.0);
}
