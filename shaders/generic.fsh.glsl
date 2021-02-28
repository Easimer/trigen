out vec4 vFrag;

#ifdef GENERIC_SHADER_WITH_VERTEX_COLORS
in vec3 vColor;
#endif

#if TEXTURED
in vec2 vUV;
uniform sampler2D texDiffuse;
#endif

void main() {
#ifdef GENERIC_SHADER_WITH_VERTEX_COLORS
    vFrag = vec4(vColor, 1.0f);
#else
#if TEXTURED
    vFrag = texture(texDiffuse, vUV);
#else
    vFrag = vec4(1.0f, 0.0f, 0.0f, 1.0f);
#endif
#endif
}