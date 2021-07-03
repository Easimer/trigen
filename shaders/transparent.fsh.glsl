#ifndef VAO_LAYOUT
#define VAO_LAYOUT(i) layout (location = i)
#endif

VAO_LAYOUT(0) out vec4 outBaseColor;

in vec2 vUV;
uniform sampler2D texDiffuse;
uniform vec4 tintColor;

void main() {
    vec4 diffuse = texture(texDiffuse, vUV);
    diffuse.a = floor(diffuse.a);
    if(diffuse.a < 1) {
        discard;
    }
    outBaseColor = diffuse * tintColor;
}
