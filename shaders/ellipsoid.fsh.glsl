// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: Ellipsoid raymarching pixel shader
//

#version 330 core

// Distance to the near clipping plane
#define NEAR_CLIPPING_PLANE 0.1
// Distance to the far clipping plane
#define FAR_CLIPPING_PLANE 1000.0
// Number of raymarching steps
#define STEPS_N 64
// Epsilon value
#define EPSILON 0.01
// Distance bias
#define DISTANCE_BIAS 0.7
// Set this to 1 to return early during raymarching steps if the
// ray gets too far from or too near to the camera
// TODO(danielm): not sure which value is better, need to
// profile this somehow.
#define RETURN_EARLY 0

// Screen coordinates, x,y in [-1, 1]
in vec2 vUV;
// Fragment color
out vec4 vFrag;

// Particle position
uniform vec3 vTranslation;
// Particle rotation
uniform mat3 matRotation;
// Particle size
uniform vec3 vSize;
// Particle color
const vec4 vColor = vec4(0.6, 0.6, 0.6, 1.0);

// View-projection matrix and it's inverse
uniform mat4 matVP;
uniform mat4 matInvVP;

// Position of the sun
uniform vec3 vSun;

/**
 * Ellipsoid distance function
 * @param p Sample point
 * @param r Ellipsoid radii
 * @return Distance from the surface of the ellipsoid defined by the above
 * parameters
 */
float sdEllipsoid(vec3 p, vec3 r) {
  float k0 = length(p / r);
  float k1 = length(p / (r * r));
  return k0 * (k0 - 1.0) / k1;
}

/**
 * Scene distance function
 * @param p Sample point
 * @return Distance from the surface of the scene
 */
float scene(vec3 p) {
	// Transform the sample point into model space
	vec3 sp = matRotation * (p - vTranslation);
	return sdEllipsoid(sp, vSize);
}

/**
 * Calculate the normal at a given intersection point.
 * @param ray_hit_position the intersect point
 * @param smoothness surface smoothness, [0.0, 1.0]
 * @return surface normal
 */
vec3 normal(vec3 ray_hit_position, float smoothness) {
	vec3 n;
	vec2 dn = vec2(smoothness, 0.0);
	n.x	= scene(ray_hit_position + dn.xyy) - scene(ray_hit_position - dn.xyy);
	n.y	= scene(ray_hit_position + dn.yxy) - scene(ray_hit_position - dn.yxy);
	n.z	= scene(ray_hit_position + dn.yyx) - scene(ray_hit_position - dn.yyx);
	return normalize(n);
}

// Ray descriptor
struct Ray {
	vec3 eye;
	vec3 dir;
};

/**
 * Calculate the origin and direction of the ray from the
 * view-projection matrix.
 * @return Ray origin and direction
 */
Ray getRay() {
    vec4 near = vec4(vUV, 0.0, 1.0);
	near = matInvVP * near;
    vec4 far = near + matInvVP[2];
    near.xyz /= near.w;
    far.xyz /= far.w;
    return Ray(near.xyz, normalize(far.xyz-near.xyz));
}

/**
 * Calculate the fragment depth value given the intersection point.
 * @param intersect Intersection point
 * @return Fragment depth value (write it to gl_FragDepth)
 */
float getFragmentDepth(vec3 intersect) {
	float zc = (matVP * vec4(intersect, 1.0)).z;
	float wc = (matVP * vec4(intersect, 1.0)).w;
	return zc / wc;
}

void main() {
	Ray r = getRay();

	float dist = NEAR_CLIPPING_PLANE;

	for(int i = 0; i < STEPS_N; i++) {
		vec3 p = r.eye + dist * r.dir;
		float temp = scene(p);
		#if RETURN_EARLY
		if(temp < EPSILON) {
			break;
		}
		#endif /* RETURN_EARLY */

		dist += temp * DISTANCE_BIAS;

		#if RETURN_EARLY
		if(dist > FAR_CLIPPING_PLANE) {
			break;
		}
		#endif /* RETURN_EARLY */
	}

	vec3 intersect = r.eye + dist * r.dir;

	gl_FragDepth = getFragmentDepth(intersect);

	vec3 sunDir = normalize(vSun - vTranslation);
	vec3 normal = normal(intersect, 1);
	float illum = min(max(0.2, dot(normal, sunDir)), 1.0);
	vFrag = illum * vColor;

	if(!(NEAR_CLIPPING_PLANE < dist && dist < FAR_CLIPPING_PLANE)) {
		// Ray went beyond the far plane
		// TODO(danielm): make this discard go away
		discard;
	}
}