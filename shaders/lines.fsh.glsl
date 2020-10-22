in float t;
out vec4 vFrag;

uniform vec3 vColor0, vColor1;

void main() {
    // vec3 col0 = vec3(0, 1, 0);
    // vec3 col1 = vec3(1, 0, 1);
    vec3 col0 = vColor0;
    vec3 col1 = vColor1;
    vFrag = vec4((1 - t) * col0 + t * col1, 1.0);
}
