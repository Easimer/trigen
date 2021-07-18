// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: deferred rendering lighting pass
//

uniform sampler2D texBaseColor;
uniform sampler2D texNormal;
uniform sampler2D texPosition;

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

void main() {
    vec4 baseColorRGBA = texture(texBaseColor, vUV).rgba;
    vec3 baseColor = baseColorRGBA.rgb;
    if(baseColorRGBA.a  == 0) {
        discard;
    }
    vec3 normal = texture(texNormal, vUV).rgb;
    vec3 position = texture(texPosition, vUV).rgb;

    vec3 viewDir = normalize(viewPosition.xyz - position);
    vec3 surfaceColor = 0.1 * baseColor;

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

        surfaceColor += diffuse + specular;
    }

    // Gamma-correction
    surfaceColor = pow(surfaceColor, vec3( 1 / 2.2));

    vFrag = vec4(surfaceColor, 1);
}