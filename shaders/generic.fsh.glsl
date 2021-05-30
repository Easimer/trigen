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
// Tangent-space sun position
in vec3 tSunPosition;
// Tangent-space view position
in vec3 tViewPosition;
// Tangent-space fragment position
in vec3 tFragPosition;
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
    vec3 normal = texture(texNormal, vUV).rgb;
    normal = normalize(normal * 2.0 - 1.0);

    vec3 lightDir   = normalize(tSunPosition - tFragPosition);
    vec3 viewDir    = normalize(tViewPosition - tFragPosition);
    vec3 halfwayDir = normalize(lightDir + viewDir);
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diff * vec3(1, 1, 1);

    vFrag = vec4(diffuse * baseColor, 1);
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
