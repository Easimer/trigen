#ifndef VAO_LAYOUT
#define VAO_LAYOUT(i) layout (location = i)
#endif

VAO_LAYOUT(0) out vec4 outBaseColor;
VAO_LAYOUT(1) out vec4 outNormal;
VAO_LAYOUT(2) out vec4 outPosition;

in vec3 vPosition;

#ifdef GENERIC_SHADER_WITH_VERTEX_COLORS
in vec3 vColor;
#endif

#if TEXTURED
in vec2 vUV;
uniform sampler2D texDiffuse;
uniform sampler2D texNormal;
#endif

#if LIT
in mat3 vTBN;
#endif /* LIT */

uniform vec4 tintColor;

#if GENERIC_SHADER_WITH_VERTEX_COLORS
void main() {
    outBaseColor = vec4(1, 1, 1, 1.0f) * tintColor;
    // outBaseColor = vec4(vColor, 1.0f) * tintColor;
    outNormal = vec4(0, 0, 1, 1);
    outPosition = vec4(vPosition, 1);
}
#else /* GENERIC_SHADER_WITH_VERTEX_COLORS */

#if TEXTURED
#if LIT
void main() {
    // Sample textures
    vec3 normal = texture(texNormal, vUV).rgb;
    normal = normalize(normal * 2.0 - 1.0);
    normal = vTBN * normal;

    outBaseColor = texture(texDiffuse, vUV);
    outNormal = vec4(normal, 1);
    outPosition = vec4(vPosition, 1);
}
#else /* LIT */
void main() {
    outBaseColor = texture(texDiffuse, vUV) * tintColor;
    outNormal = vec4(0, 0, 1, 1);
    outPosition = vec4(vPosition, 1);
}
#endif /* LIT */
#else /* TEXTURED */
void main() {
    outBaseColor = vec4(0.828125f, 0.828125f, 0.828125f, 1.0f) * tintColor;
    outNormal = vec4(0, 0, 1, 1);
    outPosition = vec4(vPosition, 1);
}
#endif /* TEXTURED */
#endif /* GENERIC_SHADER_WITH_VERTEX_COLORS */
