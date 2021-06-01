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

out vec3 vPosition;

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

out mat3 vTBN;

#endif /* LIT */

void main() {
#ifdef GENERIC_SHADER_WITH_VERTEX_COLORS
    vColor = vec3(aColor.r / 255.0f, aColor.g / 255.0f, aColor.b / 255.0f);
#endif

    gl_Position = matMVP * vec4(aPosition.xyz, 1.0);
    vPosition = gl_Position.xyz;
#if TEXTURED
    vUV = vec2(aUV.x, 1 - aUV.y);
#if LIT
    mat3 matModelRot = mat3(matModel);
    vec3 T = normalize(matModelRot * aTangent);
    vec3 B = normalize(matModelRot * aBitangent);
    vec3 N = normalize(matModelRot * aNormal);
    vTBN = mat3(T, B, N);
#endif
#endif
}