#ifndef VAO_LAYOUT
#define VAO_LAYOUT(i) layout (location = i)
#endif

VAO_LAYOUT(0) in vec3 aPosition;

#ifdef GENERIC_SHADER_WITH_VERTEX_COLORS
VAO_LAYOUT(1) in vec3 aColor;
#endif

#if TEXTURED
VAO_LAYOUT(1) in vec2 aUV;
out vec2 vUV;
#endif

uniform mat4 matMVP;

#ifdef GENERIC_SHADER_WITH_VERTEX_COLORS
out vec3 vColor;
#endif

#if LIT
uniform vec3 sunPosition;
uniform mat3 matModel;
out vec3 vSunDirection;
out mat3 vTBNMatrix;
out vec3 vNormal;

VAO_LAYOUT(2) in vec3 aNormal;
VAO_LAYOUT(3) in vec3 aTangent;
VAO_LAYOUT(4) in vec3 aBitangent;
#endif /* LIT */

void main() {
#ifdef GENERIC_SHADER_WITH_VERTEX_COLORS
    vColor = vec3(aColor.r / 255.0f, aColor.g / 255.0f, aColor.b / 255.0f);
#endif

    gl_Position = matMVP * vec4(aPosition.xyz, 1.0);
#if TEXTURED
    vUV =  vec2(aUV.x, 1 - aUV.y);
#if LIT
    vec3 T = normalize(matModel * aTangent);
    vec3 B = normalize(matModel * aBitangent);
    vec3 N = normalize(matModel * aNormal);
    vTBNMatrix = mat3(T, B, N);
    vSunDirection = normalize(-sunPosition);
#endif
#endif
}