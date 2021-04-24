out vec4 vFrag;

#ifdef GENERIC_SHADER_WITH_VERTEX_COLORS
in vec3 vColor;
#endif

#if TEXTURED
in vec2 vUV;
uniform sampler2D texDiffuse;
#endif

uniform vec4 tintColor;

void main() {
#ifdef GENERIC_SHADER_WITH_VERTEX_COLORS
    vFrag = vec4(vColor, 1.0f) * tintColor;
#else
#if TEXTURED
    vFrag = texture(texDiffuse, vec2(vUV.x, 1 - vUV.y)) * tintColor;
#else
    vFrag = vec4(0.828125f, 0.828125f, 0.828125f, 1.0f) * tintColor;
#endif
#endif
}