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
VAO_LAYOUT(2) in vec3 aNormal;
VAO_LAYOUT(3) in vec3 aTangent;
VAO_LAYOUT(4) in vec3 aBitangent;

// World position of the sun
uniform vec3 sunPosition;
uniform mat4 matModel;
uniform vec3 viewPosition;

// Tangent-space sun position
out vec3 tSunPosition;
// Tangent-space view position
out vec3 tViewPosition;
// Tangent-space fragment position
out vec3 tFragPosition;

#endif /* LIT */

void main() {
#ifdef GENERIC_SHADER_WITH_VERTEX_COLORS
    vColor = vec3(aColor.r / 255.0f, aColor.g / 255.0f, aColor.b / 255.0f);
#endif

    gl_Position = matMVP * vec4(aPosition.xyz, 1.0);
#if TEXTURED
    vUV = vec2(aUV.x, 1 - aUV.y);
#if LIT
    mat3 matModelRot = mat3(matModel);
    vec3 T = normalize(matModelRot * aTangent);
    vec3 B = normalize(matModelRot * aBitangent);
    vec3 N = normalize(matModelRot * aNormal);
    mat3 TBN = transpose(mat3(T, B, N));
    
    tSunPosition = TBN * sunPosition;
    tViewPosition = TBN * viewPosition;
    tFragPosition = TBN * vec3(matModel * vec4(aPosition.xyz, 1.0));
#endif
#endif
}