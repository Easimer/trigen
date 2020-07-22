#version 330 core

#define NEAR_CLIPPING_PLANE 0.1
#define FAR_CLIPPING_PLANE 1000.0
#define STEPS_N 40
#define EPSILON 0.01
#define DISTANCE_BIAS 0.7

in vec4 vUV;
out vec4 vFrag;

uniform vec3 vTranslation;
uniform mat3 matRotation;

uniform vec3 vCamTranslation;
uniform mat3 matCamRotation;

uniform vec3 vSize;

uniform mat4 matView;
uniform mat4 matProj;
uniform mat4 matMVP;

const vec4 vSun = vec4(10, 10, 10, 0);
const vec4 vColor = vec4(0.6, 0.6, 0.6, 1.0);

float sdfSphere(vec3 p, float r) {
	return length(p) - r;
}

float sdEllipsoid(vec3 p, vec3 r) {
  float k0 = length(p / r);
  float k1 = length(p / (r * r));
  return k0 * (k0 - 1.0) / k1;
}

float sdBox(vec3 p, vec3 b) {
  vec3 q = abs(p) - b;
  return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0);
}

// TODO:
// - cache MVP inverse
// - sun

vec2 scene(vec3 p) {
	vec3 sp = matRotation * (p - vTranslation); // sample point in model space
	return vec2(sdEllipsoid(sp, vSize * 0.125f), 0);
}

vec3 normal(vec3 ray_hit_position, float smoothness) {
	vec3 n;
	vec2 dn = vec2(smoothness, 0.0);
	n.x	= scene(ray_hit_position + dn.xyy).x - scene(ray_hit_position - dn.xyy).x;
	n.y	= scene(ray_hit_position + dn.yxy).x - scene(ray_hit_position - dn.yxy).x;
	n.z	= scene(ray_hit_position + dn.yyx).x - scene(ray_hit_position - dn.yyx).x;
	return normalize(n);
}

float map_dist_to_alpha(float dist) {
	float alpha = 1;

	return alpha;
}

struct Ray {
	vec3 eye;
	vec3 dir;
};

Ray get_ray() {
    vec4 near = vec4(vUV.xy, 0.0, 1.0);
	mat4 matInverseMVP = inverse(matMVP);
    near = matInverseMVP * near;
    vec4 far = near + matInverseMVP[2];
    near.xyz /= near.w;
    far.xyz /= far.w;
    return Ray(near.xyz, far.xyz-near.xyz);
}

#define EARLY_RETURN 0

void main() {
	Ray r = get_ray();
	vec3 eye = r.eye;
	vec3 worldDir = normalize(r.dir);

	float dist = NEAR_CLIPPING_PLANE;

	for(int i = 0; i < STEPS_N; i++) {
		vec3 p = eye + dist * worldDir;
		float temp = scene(p).x;
		#if EARLY_RETURN
		if(temp < EPSILON) {
			break;
		}
		#endif /* EARLY_RETURN */

		dist += temp * DISTANCE_BIAS;

		#if EARLY_RETURN
		if(dist > FAR_CLIPPING_PLANE) {
			break;
		}
		#endif /* EARLY_RETURN */
	}

	vec4 intersect = vec4(eye + dist * worldDir, 1);

	//float zc = (matProj * matView * vec4(intersect.xyz, 1.0)).z;
	//float wc = (matProj * matView * vec4(intersect.xyz, 1.0)).w;
	float zc = (matMVP * vec4(intersect.xyz, 1.0)).z;
	float wc = (matMVP * vec4(intersect.xyz, 1.0)).w;
	gl_FragDepth = zc/wc;

	vec3 sunDir = normalize(vSun.xyz - vTranslation.xyz);
	vec3 normal = normal(intersect.xyz, 1);
	float illum = min(max(0.2, dot(normal, sunDir)), 1.0);
	vFrag = illum * vColor;

	if(!(NEAR_CLIPPING_PLANE < dist && dist < FAR_CLIPPING_PLANE)) {
		// Ray went beyond the far plane
		discard;
	}
}
