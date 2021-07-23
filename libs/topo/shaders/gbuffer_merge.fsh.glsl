// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: deferred rendering lighting pass
//

uniform sampler2D texBaseColor;
uniform sampler2D texNormal;
uniform sampler2D texPosition;
uniform sampler2D texShadowMap;

uniform mat4 shadowCasterViewProj;

in vec2 vUV;

out vec4 vFrag;

struct Light {
    vec4 position;
    vec4 color;
};

layout(std140, binding = 0) buffer Lights {
    vec4 viewPosition;
    int numLights;

    Light lights[];
};

float calculateShadow(vec4 worldPosition) {
    vec4 projCoords = (shadowCasterViewProj * worldPosition);
    projCoords /= projCoords.w;
    projCoords = projCoords * 0.5 + 0.5;

    float closestDepth = texture(texShadowMap, projCoords.xy).r;
    float currentDepth = projCoords.z;
    float bias = 0.005;
    return currentDepth - bias > closestDepth  ? 1.0 : 0.0;
}

void main() {
    vec4 baseColorRGBA = texture(texBaseColor, vUV).rgba;
    vec3 baseColor = baseColorRGBA.rgb;
    if(baseColorRGBA.a  == 0) {
        discard;
    }
    vec3 normal = texture(texNormal, vUV).rgb;
    vec3 position = texture(texPosition, vUV).rgb;

    vec3 viewDir = normalize(viewPosition.xyz - position);

    vec3 surfaceDiffuse = vec3(0, 0, 0);
    vec3 surfaceSpecular = vec3(0, 0, 0);

    for(int i = 0; i < numLights; i++) {
        vec3 lightColor = lights[i].color.rgb;
        vec3 lightDir = normalize(lights[i].position.xyz - position);
        float dist = length(lightDir);

        float attenuation = lights[i].color.a / (dist * dist);
        vec3 diffuse = attenuation * max(dot(normal, lightDir), 0) * baseColor * lightColor;

        vec3 halfwayDir = normalize(lightDir + viewDir);
        float spec = pow(max(dot(normal, halfwayDir), 0.0), 32.0);
        // vec3 specular = lightColor * spec;
        vec3 specular = vec3(0, 0, 0);

        surfaceDiffuse += diffuse;
        surfaceSpecular += specular;
    }

    float shadow = calculateShadow(vec4(position, 1));
    vec3 surfaceColor = (0.1 * baseColor + (1.0 - shadow) * (surfaceDiffuse + surfaceSpecular)) * baseColor;
    // vec3 surfaceColor = vec3(1 - shadow, 1 - shadow, 1 - shadow);

    // Gamma-correction
    surfaceColor = pow(surfaceColor, vec3( 1 / 2.2));

    vFrag = vec4(surfaceColor, 1);
}