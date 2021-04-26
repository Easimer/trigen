out vec4 vFrag;

#ifdef GENERIC_SHADER_WITH_VERTEX_COLORS
in vec3 vColor;
#endif

#if TEXTURED
in vec2 vUV;
uniform sampler2D texDiffuse;
uniform sampler2D texNormal;
#endif

#if LIT
in vec3 vSunDirection;
in mat3 vTBNMatrix;
#endif /* LIT */

uniform vec4 tintColor;

#if GENERIC_SHADER_WITH_VERTEX_COLORS
void main() {
    vFrag = vec4(vColor, 1.0f) * tintColor;
}
#else /* GENERIC_SHADER_WITH_VERTEX_COLORS */

#if TEXTURED
#if LIT
void main() {
    vec3 baseColor = texture(texDiffuse, vUV).rgb;
    vFrag = vec4(baseColor, 1);
    // TODO(danielm): lighting
}
#else /* LIT */
void main() {
    vFrag = texture(texDiffuse, vUV) * tintColor;
}
#endif /* LIT */
#else /* TEXTURED */
void main() {
    vFrag = vec4(0.828125f, 0.828125f, 0.828125f, 1.0f) * tintColor;
}
#endif /* TEXTURED */
#endif /* GENERIC_SHADER_WITH_VERTEX_COLORS */
