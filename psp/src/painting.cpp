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
#include <intersect.h>
#include <boost/gil.hpp>
#include <mutex>
#include <glm/geometric.hpp>
#include <glm/mat3x3.hpp>
#include <glm/trigonometric.hpp>
#include <glm/gtc/constants.hpp>
#include <random>
#include <worker_group.hpp>

/*
    :SphericalCoordinateConvention
    Here we use spherical coordinates (θ,ϕ), where θ is the angle down from the
    pole, and ϕ is the angle around the axis through the pole.
    
    x=cos(ϕ)cos(θ)
    y=sin(ϕ)cos(θ)
    z=sin(θ)
    
    u=ϕ/2π
    v=θ/π
*/

using Pixel1 = boost::gil::gray8c_pixel_t;
using Pixel3 = boost::gil::rgb8_pixel_t;

struct Out_Material {
    glm::vec<2, int> dim;
    std::unique_ptr<Pixel3[]> base;
    std::unique_ptr<Pixel3[]> normal;
    std::unique_ptr<Pixel3[]> height;
    std::unique_ptr<Pixel3[]> roughness;
    std::unique_ptr<Pixel3[]> ao;
};

struct Particle_System {
    std::vector<glm::vec3> position;
    std::vector<glm::vec3> next_position;
    std::vector<glm::vec3> velocity;
    std::vector<Pixel3> color_base;
    std::vector<Pixel3> color_normal;
    std::vector<Pixel3> color_height;
    std::vector<Pixel3> color_roughness;
    std::vector<Pixel3> color_ao;
    std::vector<bool> alive;
};

template<typename T>
static void fill_buffer(std::unique_ptr<T[]> &buffer, T value, size_t size) {
    for (size_t i = 0; i < size; i++) {
        buffer[i] = value;
    }
}

static boost::gil::rgb8_view_t make_view(PSP::Texture const &image) {
    // TODO(danielm): discarding const qualifier here!
    auto it = (boost::gil::rgb8_pixel_t *)image.buffer;
    return boost::gil::interleaved_view(image.width, image.height, it, std::ptrdiff_t(image.width * sizeof(glm::u8vec3)));
}

static boost::gil::rgb8_view_t make_view(Out_Material const &material, std::unique_ptr<Pixel3[]> const &image) {
    // TODO(danielm): discarding const qualifier here!
    auto it = (boost::gil::rgb8_pixel_t *)image.get();
    return boost::gil::interleaved_view(material.dim.x, material.dim.y, it, std::ptrdiff_t(material.dim.x * sizeof(glm::u8vec3)));
}

static boost::gil::rgb8_pixel_t sample_image(boost::gil::rgb8_view_t const &view, glm::vec2 uv) {
    auto x = ptrdiff_t(uv.x * view.width());
    // TODO(danielm): check whether we need to flip Y here:
    auto y = ptrdiff_t(uv.y * view.height());
    return view(x, y);
}

static boost::gil::rgb8_pixel_t& sample_image_ref(boost::gil::rgb8_view_t &view, glm::vec2 uv) {
    auto x = ptrdiff_t(uv.x * view.width());
    // TODO(danielm): check whether we need to flip Y here:
    auto y = ptrdiff_t(uv.y * view.height());
    return view(x, y);
}

static float coplanarity_test(glm::vec3 const &v0, glm::vec3 const &v1, glm::vec3 const &v2, glm::vec3 const &p) {
    return dot(cross(v1 - v0, p - v0), v2 - v0);
}

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

static bool parametric_form_of_point_on_triangle(glm::vec3 const &v0, glm::vec3 const &v1, glm::vec3 const &v2, glm::vec3 const &p, float &s, float &t) {
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

template<typename T>
static T interpolate(T const &v0, T const &v1, T const &v2, float s, float t) {
    auto d0 = v2 - v0;
    auto d1 = v1 - v0;

    return v0 + s * d0 + t * d1;
}

static bool is_inside_bounding_box(glm::vec3 const &min, glm::vec3 const &max, glm::vec3 const &x) {
    for (int i = 0; i < 3; i++) {
        if (x[i] < min[i] || x[i] > max[i]) {
            return false;
        }
    }
    return true;
}

// Convert a point on a given sphere to spherical coordinates.
// 
// \returns (theta, phi)
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

static bool is_same_side(glm::vec3 p1, glm::vec3 p2, glm::vec3 a, glm::vec3 b) {
    auto cp1 = cross(b - a, p1 - a);
    auto cp2 = cross(b - a, p2 - a);
    return dot(cp1, cp2) >= 0;
}

static bool is_point_in_triangle(glm::vec2 x, glm::vec2 p0, glm::vec2 p1, glm::vec2 p2) {
    auto a = glm::vec3(p0, 0.f);
    auto b = glm::vec3(p1, 0.f);
    auto c = glm::vec3(p2, 0.f);
    auto p = glm::vec3(x, 0.f);
    return (is_same_side(p, a, b, c) && is_same_side(p, b, a, c) && is_same_side(p, c, a, b));
}

static bool find_triangle_containing_uv(PSP::Mesh const *mesh, glm::vec2 uv, size_t *triangle_id) {
    assert(mesh != NULL);
    assert(0 <= uv.x && uv.x <= 1);
    assert(0 <= uv.y && uv.y <= 1);
    assert(triangle_id != NULL);

    bool ret = false;

    auto N = mesh->elements.size() / 3;
    for (size_t t = 0; t < N; t++) {
        auto i0 = mesh->elements[t * 3 + 0];
        auto i1 = mesh->elements[t * 3 + 1];
        auto i2 = mesh->elements[t * 3 + 2];

        auto uv0 = mesh->uv[t * 3 + 0];
        auto uv1 = mesh->uv[t * 3 + 1];
        auto uv2 = mesh->uv[t * 3 + 2];

        auto p0 = mesh->position[i0];
        auto p1 = mesh->position[i1];
        auto p2 = mesh->position[i2];

        auto area = 0.5f * glm::abs(glm::determinant(glm::mat3(glm::vec3(uv0, 1), glm::vec3(uv1, 1), glm::vec3(uv2, 1))));

        if (area > 0 && is_point_in_triangle(uv, uv0, uv1, uv2)) {
            // assert(ret == false);
            ret = true;
            *triangle_id = t;
//#ifdef NDEBUG
            break;
//#endif
        }
    }

    return ret;
}

class Painter : public PSP::IPainter {
public:
    Painter(PSP::Parameters const &params) : _mesh(params.mesh), _in_material(params.material) {
        auto size = params.out_width * params.out_height;
        _out_material.dim.x = params.out_width;
        _out_material.dim.y = params.out_height;
        _out_material.base = std::make_unique<Pixel3[]>(size);
        _out_material.normal = std::make_unique<Pixel3[]>(size);
        _out_material.height = std::make_unique<Pixel3[]>(size);
        _out_material.roughness = std::make_unique<Pixel3[]>(size);
        _out_material.ao = std::make_unique<Pixel3[]>(size);

        _view_base = make_view(params.material->base);
        _view_normal = make_view(params.material->normal);
        _view_height = make_view(params.material->height);
        _view_roughness = make_view(params.material->roughness);
        _view_ao = make_view(params.material->ao);

        auto bbox_min = glm::vec3(FLT_MAX, FLT_MAX, FLT_MAX);
        auto bbox_max = glm::vec3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
        for (auto &vtx : _mesh->position) {
            bbox_min = glm::min(bbox_min, vtx);
            bbox_max = glm::max(bbox_max, vtx);
        }

        auto bbox_extent = bbox_max - bbox_min;
        _radius = glm::max(bbox_extent.x, glm::max(bbox_extent.y, bbox_extent.z)) / 2;
        _center_of_mass = bbox_min + bbox_extent / 2.f;
    }

    ~Painter() override = default;

    void step_painting(float dt) override {
        Worker_Group workers;

        struct Ray {
            glm::vec3 origin;
            glm::vec3 direction;
            // Which pixel to paint
            glm::vec<2, int> texcoord;
        };

        std::vector<std::vector<Ray>> raygen_results;
        std::mutex lock_raygen_results;

        auto task = [&](int y) {
            std::vector<Ray> rays;

            auto v = 1.0f - float(y) / float(_out_material.dim.y);
            for (int x = 0; x < _out_material.dim.x; x++) {
                auto u = float(x) / float(_out_material.dim.x);

                auto uv = glm::vec2(u, v);
                size_t triangle_id;
                if (find_triangle_containing_uv(_mesh, uv, &triangle_id)) {
                    auto i0 = _mesh->elements[triangle_id * 3 + 0];
                    auto i1 = _mesh->elements[triangle_id * 3 + 1];
                    auto i2 = _mesh->elements[triangle_id * 3 + 2];
                    auto uv0 = _mesh->uv[triangle_id * 3 + 0];
                    auto uv1 = _mesh->uv[triangle_id * 3 + 1];
                    auto uv2 = _mesh->uv[triangle_id * 3 + 2];
                    auto x0 = _mesh->position[i0];
                    auto x1 = _mesh->position[i1];
                    auto x2 = _mesh->position[i2];
                    auto n0 = _mesh->normal[i0] ;
                    auto n1 = _mesh->normal[i1];
                    auto n2 = _mesh->normal[i2];

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

        for (int y = 0; y < _out_material.dim.y; y++) {
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

        glm::vec3 center_of_mass;
        float radius;
        calculate_bounding_sphere(_mesh, center_of_mass, radius);
        
        // Create views into output textures
        auto view_out_base = make_view(_out_material, _out_material.base);
        auto view_out_normal = make_view(_out_material, _out_material.normal);
        auto view_out_height = make_view(_out_material, _out_material.height);
        auto view_out_roughness = make_view(_out_material, _out_material.roughness);
        auto view_out_ao = make_view(_out_material, _out_material.ao);

        // Raycast

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

            // Gather pixel copy operations
            struct Image_Write_Request {
                glm::vec2 input_uv_coord;
                boost::gil::point_t output_pix_coord;
            };
            std::vector<Image_Write_Request> image_write_requests;
            
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

                image_write_requests.push_back({ input_uv_coord, boost::gil::point_t { R.texcoord.x, R.texcoord.y } });
                
            }

            // Copy pixels
            for (auto &iwr : image_write_requests) {
                view_out_base(iwr.output_pix_coord) = sample_image_ref(_view_base, iwr.input_uv_coord);
                view_out_normal(iwr.output_pix_coord) = sample_image_ref(_view_normal, iwr.input_uv_coord);
                view_out_height(iwr.output_pix_coord) = sample_image_ref(_view_height, iwr.input_uv_coord);
                view_out_roughness(iwr.output_pix_coord) = sample_image_ref(_view_roughness, iwr.input_uv_coord);
                view_out_ao(iwr.output_pix_coord) = sample_image_ref(_view_ao, iwr.input_uv_coord);
            }
        };

        auto const raycast_worksize = 2048;
        auto raycast_work_remains = total_rays;
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

    bool is_painting_done() override {
        return true;
    }

    size_t num_particles() override {
        return 0;
    }

    bool get_particle(size_t id, glm::vec3 *out_position, glm::vec3 *out_next_position, glm::vec3 *out_velocity) override {
        return false;
    }

    void result(PSP::Material *out_material) override {
#define COPY_TEXTURE_OUT(name) copy_texture_out(out_material->name, _out_material.dim, _out_material.name)
        COPY_TEXTURE_OUT(base);
        COPY_TEXTURE_OUT(normal);
        COPY_TEXTURE_OUT(height);
        COPY_TEXTURE_OUT(roughness);
        COPY_TEXTURE_OUT(ao);
    }

    void copy_texture_out(PSP::Texture &dst, glm::ivec2 const &dim, std::unique_ptr<Pixel3[]> &src) {
        dst.width = dim.x;
        dst.height = dim.y;
        dst.buffer = src.get();
    }

private:
    PSP::Mesh const *_mesh;
    PSP::Material const *_in_material;
    Out_Material _out_material;

    boost::gil::rgb8_view_t _view_base;
    boost::gil::rgb8_view_t _view_normal;
    boost::gil::rgb8_view_t _view_height;
    boost::gil::rgb8_view_t _view_roughness;
    boost::gil::rgb8_view_t _view_ao;

    float _radius;
    glm::vec3 _center_of_mass;
};

std::unique_ptr<PSP::IPainter> PSP::make_painter(PSP::Parameters const &params) {
    return std::make_unique<Painter>(params);
}
