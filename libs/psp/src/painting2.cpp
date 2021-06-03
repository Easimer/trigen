// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: 
//

#include "painting.h"

#include <cfloat>
#include <cmath>
#include <memory>
#include <vector>
#include <thread>
#include <random>
#include <mutex>

#include <intersect.h>
#include <worker_group.hpp>

#include <glm/geometric.hpp>
#include <glm/mat3x3.hpp>
#include <glm/trigonometric.hpp>
#include <glm/gtc/constants.hpp>

#include <boost/gil.hpp>

#include "uv_quadtree.h"

#include <Tracy.hpp>

using Input_View = boost::gil::rgb8_view_t;
using Output_View = boost::gil::rgb8_view_t;

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
    // Which pixel to paint
    glm::vec<2, int> texcoord;
};

/**
 * \brief Used to test for whether a point is in the same plane as a triangle.
 * \param v0 1st vertex of the triangle
 * \param v1 2nd vertex of the triangle
 * \param v2 3rd vertex of the triangle
 * \param p Point to be tested
 * \return Returns zero when the point is in the triangle's plane and a
 * non-zero value otherwise.
 */
static float coplanarity_test(
        glm::vec3 const &v0,
        glm::vec3 const &v1,
        glm::vec3 const &v2,
        glm::vec3 const &p) {
    return dot(cross(v1 - v0, p - v0), v2 - v0);
}

/**
 * \brief Calculates the parametric coordinates of a 3D point w.r.t. a given
 * triangle.
 * \param v0 1st vertex of the triangle
 * \param v1 2nd vertex of the triangle
 * \param v2 3rd vertex of the triangle
 * \param p The point
 * \param [in] s S-coordinate of the point
 * \param [in] t T-coordinate of the point
 * \return Returns whether the point is coplanar with the triangle.
 */
static bool parametric_form_of_point_on_triangle(
        glm::vec3 const &v0,
        glm::vec3 const &v1,
        glm::vec3 const &v2,
        glm::vec3 const &p,
        float &s,
        float &t) {
    // check for coplanarity
    if (coplanarity_test(v0, v1, v2, p) != 0) {
        return false;
    }

    auto b0 = v2 - v0;
    auto b1 = v1 - v0;
    // Find `s` and `t` such that `p = v0 + s * b0 + t * b1`

    auto denominator = 1.0f / (b0.x * b1.y - b0.y * b1.x);
    // Should only happen when v0, v1 and v2 are the same point.
    assert(!std::isnan(denominator));

    s = (p.x * b1.y - p.y * b1.x - v0.x * b1.y + v0.y * b1.x) * denominator;
    t = (-p.x * b0.y + p.y * b0.x + v0.x * b0.y - v0.y * b0.x) * denominator;

    return true;
}

/**
 * \brief Calculates the bounding sphere of the mesh.
 * \param [in] mesh Mesh
 * \param [out] center_of_mass Center of the sphere
 * \param [out] radius Radius of the sphere
 */
static void calculate_bounding_sphere(PSP::Mesh const *mesh, glm::vec3 &center_of_mass, float &radius) {
    auto bbox_min = glm::vec3(FLT_MAX, FLT_MAX, FLT_MAX);
    auto bbox_max = glm::vec3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    for (auto &vtx : mesh->position) {
        bbox_min = glm::min(bbox_min, vtx);
        bbox_max = glm::max(bbox_max, vtx);
    }

    auto bbox_extent = bbox_max - bbox_min;
    radius = glm::max(bbox_extent.x, glm::max(bbox_extent.y, bbox_extent.z)) / 2 * 1.5f;
    center_of_mass = bbox_min + bbox_extent / 2.f;
}

/**
 * \brief Make a view into an input texture.
 * \param image Input texture
 * \return The view
 */
static boost::gil::rgb8_view_t make_view(PSP::Input_Texture const &image) {
    // TODO(danielm): discarding const qualifier here!
    auto it = (boost::gil::rgb8_pixel_t *)image.buffer;
    return boost::gil::interleaved_view(image.width, image.height, it, std::ptrdiff_t(image.width * sizeof(glm::u8vec3)));
}

/**
 * \brief Make a view into an output texture.
 * \param image Output texture
 * \return The view
 */
static boost::gil::rgb8_view_t make_view(PSP::Output_Texture &image) {
    auto it = (boost::gil::rgb8_pixel_t *)image.buffer.get();
    return boost::gil::interleaved_view(image.width, image.height, it, std::ptrdiff_t(image.width * sizeof(glm::u8vec3)));
}

/**
 * \brief Convert a point on a given sphere to spherical coordinates.
 * \param p Point
 * \param center_of_mass Center of the sphere
 * \param radius Radius of the sphere
 * \return (theta, phi)
 */
static glm::vec<2, float> cartesian_to_spherical(glm::vec3 const &p, glm::vec3 const &center, float radius) {
    auto p1 = p - center;
    auto l1 = length(p1);
    assert(glm::abs(l1 - radius) < 0.01f);
    auto theta = glm::asin(p1.z / radius);
    auto cos_theta = glm::cos(theta);
    //auto phi = glm::asin(p1.y / (radius * cos_theta));
    auto phi = std::atan2(p1.y, p1.x);

    return { theta, phi };
}

/**
 * Samples the input view using UV coordinates.
 * \param view Input texture
 * \param uv UV coordinates
 * \return The color of the pixel or black if the UV coordinates are out of
 * bounds.
 */
static boost::gil::rgb8_pixel_t sample_image(Input_View const &view, glm::vec2 uv) {
    auto x = ptrdiff_t(uv.x * view.width());
    auto y = ptrdiff_t(uv.y * view.height());
    if (x < view.width() && y < view.height()) {
        return view(x, y);
    } else {
        return { 0, 0, 0 };
    }
}

namespace PSP {

/**
 * Make views into the input textures in the list.
 * \param mat A vector of input materials
 * \return A vector of input views
 */
static std::vector<Input_View> make_input_views(Input_Material const &mat) {
    std::vector<Input_View> inputViews;

    for (auto &inputTexture : mat) {
        inputViews.emplace_back(make_view(inputTexture));
    }

    return inputViews;
}

static Output_Material make_output_textures(Paint_Input const &params) {
    auto numInputTextures = params.inputMaterial.size();
    Output_Material ret;
    ret.reserve(numInputTextures);

    for(size_t outTexIdx = 0; outTexIdx < numInputTextures; outTexIdx++) {
        ret.emplace_back(Output_Texture {
            params.outWidth,
            params.outHeight,
            std::make_unique<uint8_t[]>(params.outWidth * params.outHeight * 3),
        });
    }

    return ret;
}

static std::vector<Output_View> make_output_views(Output_Material &mat) {
    std::vector<Output_View> views;

    for (auto &texture : mat) {
        views.emplace_back(make_view(texture));
    }

    return views;
}

static std::vector<Ray> raygen(
        Worker_Group &workers,
        Paint_Input const &params,
        std::unique_ptr<UVSpatialIndex> &uvIndex) {
    std::vector<std::vector<Ray>> raygen_results;
    std::mutex lock_raygen_results;
    auto const imageHeight = params.outHeight;
    auto const imageWidth = params.outWidth;
    auto const mesh = params.mesh;

    auto task = [&](int y) {
        std::vector<Ray> rays;

        auto v = 1.0f - float(y) / float(imageHeight);
        for (int x = 0; x < imageWidth; x++) {
            auto u = float(x) / float(imageWidth);

            auto uv = glm::vec2(u, v);
            auto maybe_triangle_id = uvIndex->find_triangle(uv);
            if (maybe_triangle_id.has_value()) {
                auto triangle_id = maybe_triangle_id.value();

                auto i0 = mesh->elements[triangle_id * 3 + 0];
                auto i1 = mesh->elements[triangle_id * 3 + 1];
                auto i2 = mesh->elements[triangle_id * 3 + 2];
                auto uv0 = mesh->uv[triangle_id * 3 + 0];
                auto uv1 = mesh->uv[triangle_id * 3 + 1];
                auto uv2 = mesh->uv[triangle_id * 3 + 2];
                auto x0 = mesh->position[i0];
                auto x1 = mesh->position[i1];
                auto x2 = mesh->position[i2];
                auto n0 = mesh->normal[i0] ;
                auto n1 = mesh->normal[i1];
                auto n2 = mesh->normal[i2];

                // Not we know that `uv` is contained by this
                // triangle.
                // Find out how this point is related to the
                // vertices of the triangle.
                float s, t;
                if (parametric_form_of_point_on_triangle({ uv0, 0 }, { uv1, 0 }, { uv2, 0 }, { uv, 0 }, s, t)) {
                    auto b0 = x2 - x0;
                    auto b1 = x1 - x0;
                    auto p = x0 + s * b0 + t * b1;

                    auto ray_position = p;
                    auto ray_normal = normalize((n0 + n1 + n2) / 3.0f);

                    rays.push_back({ ray_position, ray_normal, {x, y} });
                } else {
                    assert(!"triangle doesn't contain uv");
                }
            }
        }

        std::lock_guard g(lock_raygen_results);
        raygen_results.push_back(std::move(rays));
    };

    for (int y = 0; y < imageHeight; y++) {
        workers.emplace_task(Worker_Group::Task(std::bind(task, y)));
    }
    workers.wait();

    // Aggregate generated rays
    std::vector<Ray> rays;
    size_t total_rays = 0;
    for (auto &raygen_result : raygen_results) {
        total_rays += raygen_result.size();
    }
    rays.reserve(total_rays);

    for (auto &raygen_result : raygen_results) {
        rays.insert(rays.end(), raygen_result.cbegin(), raygen_result.cend());
    }

    return rays;
}

static void raycast(
    Worker_Group &workers,
    std::vector<Input_View> const &inputViews,
    std::vector<Output_View> &outputViews,
    PSP::Mesh const *mesh,
    std::vector<Ray> const &rays) {
    assert(inputViews.size() == outputViews.size());
    assert(mesh != nullptr);

    glm::vec3 center_of_mass;
    float radius;
    calculate_bounding_sphere(mesh, center_of_mass, radius);

    auto raycast_task = [&](size_t const start_index, size_t const count) {
        // Setup intersection test
        std::vector<glm::vec3> ray_origins(count);
        std::vector<glm::vec3> ray_directions(count);
        std::vector<glm::vec3> xp(count);
        std::vector<int> result(count);

        for (size_t i = 0; i < count; i++) {
            auto gid = start_index + i;
            auto &R = rays[gid];
            ray_origins[i] = R.origin;
            ray_directions[i] = R.direction;
        }

        intersect::ray_sphere(center_of_mass, radius, count, ray_origins.data(), ray_directions.data(), result.data(), xp.data());

        // Process raycast results

        // Generate pixel copy operations
        struct Pixel_Copy_Request {
            glm::vec2 input_uv_coord;
            boost::gil::point_t output_pix_coord;
        };
        std::vector<Pixel_Copy_Request> pixel_copy_requests;
        
        for (size_t i = 0; i < count; i++) {
            if (result[i] == 0) {
                // Ray missed
                continue;
            }

            size_t gid = start_index + i;
            auto &R = rays[gid];

            auto sph_coord = cartesian_to_spherical(xp[i], center_of_mass, radius);
            auto const pi = glm::pi<float>();
            auto input_uv_coord = glm::vec2(sph_coord.x / pi + 0.5f, sph_coord.y / (2 * pi) + 0.5f);

            pixel_copy_requests.push_back({ input_uv_coord, boost::gil::point_t { R.texcoord.x, R.texcoord.y } });
            
        }

        // Copy pixels
        for (auto &iwr : pixel_copy_requests) {
            for(size_t texIdx = 0; texIdx < outputViews.size(); texIdx++) {
                auto &outputView = outputViews[texIdx];
                auto &inputView = inputViews[texIdx];
                outputView(iwr.output_pix_coord) = sample_image(inputView, iwr.input_uv_coord);
            }
        }
    };

    auto const raycast_worksize = 2048;
    auto raycast_work_remains = rays.size();
    auto raycast_work_start = 0;

    while (raycast_work_remains > raycast_worksize) {
        workers.emplace_task(std::bind(raycast_task, raycast_work_start, raycast_worksize));
        raycast_work_start += raycast_worksize;
        raycast_work_remains -= raycast_worksize;
    }

    if (raycast_work_remains > 0) {
        workers.emplace_task(std::bind(raycast_task, raycast_work_start, raycast_work_remains));
    }

    workers.wait();
}

Output_Material paint(Paint_Input const &params) {
    Worker_Group workers;

    auto inputViews = make_input_views(params.inputMaterial);
    auto ret = make_output_textures(params);
    auto outputViews = make_output_views(ret);

    auto uvIndex = make_uv_spatial_index(params.mesh);
    auto rays = raygen(workers, params, uvIndex);
    raycast(workers, inputViews, outputViews, params.mesh, rays);

    return ret;
}

}
