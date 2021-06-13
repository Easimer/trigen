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
    vec3 position;
    vec3 color;
};

#ifndef NUM_MAX_LIGHTS
#error "NUM_MAX_LIGHTS was not defined!"
#endif

uniform int numLights;
uniform Light lights[NUM_MAX_LIGHTS];
uniform vec3 viewPosition;

void main() {
    vec4 baseColorRGBA = texture(texBaseColor, vUV).rgba;
    vec3 baseColor = baseColorRGBA.rgb;
    if(baseColorRGBA.a  == 0) {
        discard;
    }
    vec3 normal = texture(texNormal, vUV).rgb;
    vec3 position = texture(texPosition, vUV).rgb;

    vec3 viewDir = normalize(viewPosition - position);
    vec3 surfaceColor = 0.1 * baseColor;
    for(int i = 0; i < numLights; i++) {
        vec3 lightColor = lights[i].color;
        vec3 lightDir = normalize(lights[i].position - position);
        float dist = length(lightDir);

        float attenuation = 1.0 / (dist * dist);
        vec3 diffuse = attenuation * max(dot(normal, lightDir), 0) * baseColor * lightColor;

        vec3 halfwayDir = normalize(lightDir + viewDir);
        float spec = pow(max(dot(normal, halfwayDir), 0.0), 16.0);
        vec3 specular = lightColor * spec;

        surfaceColor += diffuse + specular;
    }

    // Gamma-correction
    surfaceColor = pow(surfaceColor, vec3( 1 / 2.2));

    vFrag = vec4(surfaceColor, 1);
}