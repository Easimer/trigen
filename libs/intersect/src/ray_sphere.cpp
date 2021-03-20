// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#include <intersect.h>
#include <glm/geometric.hpp>

namespace intersect {
	bool ray_sphere(
		glm::vec3 const &sphere_origin, float sphere_radius,
		glm::vec3 const &ray_origin, glm::vec3 const &ray_direction,
		float &t
	) {
		
		return true;
	}

	bool ray_sphere(
		glm::vec3 const &sphere_origin, float sphere_radius,
		glm::vec3 const &ray_origin, glm::vec3 const &ray_direction,
		glm::vec3 &xp
	) {
		int result = 0;

		ray_sphere(sphere_origin, sphere_radius, 1, &ray_origin, &ray_direction, &result, &xp);

		return result != 0;
	}

	void ray_sphere(
		glm::vec3 const &sphere_origin, float sphere_radius,
		size_t num_rays,
		glm::vec3 const *arr_ray_origin, glm::vec3 const *arr_ray_direction,
		int *arr_ray_result, glm::vec3 *arr_xp
	) {
		auto sphere_radius_squared = sphere_radius * sphere_radius;

		for (size_t i = 0; i < num_rays; i++) {
			auto oc = arr_ray_origin[i] - sphere_origin;
			auto a = arr_ray_direction[i].length();
			auto b = 2 * dot(oc, arr_ray_direction[i]);
			auto c = oc.length() - sphere_radius_squared;
			auto discriminant = b * b - 4 * a * c;

			if (discriminant < 0) {
				arr_ray_result[i] = 0;
			} else {
				auto t = (-b - glm::sqrt(discriminant)) / (2 * a);
				arr_xp[i] = arr_ray_origin[i] + t * arr_ray_direction[i];
				arr_ray_result[i] = 1;
			}
		}
	}
}