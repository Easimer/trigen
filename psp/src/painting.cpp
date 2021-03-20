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

class Old_Painter : public PSP::IPainter {
public:
    Old_Painter(PSP::Parameters const &params) : _mesh(params.mesh), _in_material(params.material) {
        auto size = params.out_width * params.out_height;
        _out_material.dim.x = params.out_width;
        _out_material.dim.y = params.out_height;
        _out_material.base = std::make_unique<Pixel3[]>(size);
        _out_material.normal = std::make_unique<Pixel3[]>(size);
        _out_material.height = std::make_unique<Pixel3[]>(size);
        _out_material.roughness = std::make_unique<Pixel3[]>(size);
        _out_material.ao = std::make_unique<Pixel3[]>(size);

        _particles_to_be_emitted = size_t(size / 8);

        // fill_buffer(_out_material.base, { 0, 0, 0 }, size);
        // fill_buffer(_out_material.normal, { 0, 0, 255 }, size);
        // fill_buffer(_out_material.height, { 0, 0, 0 }, size);
        // fill_buffer(_out_material.roughness, { 0, 0, 0 }, size);
        // fill_buffer(_out_material.ao, { 0, 0, 0 }, size);

        _theta_step = glm::pi<float>() / params.subdiv_theta;
        _phi_step = 2 * glm::pi<float>() / params.subdiv_phi;

        _view_base = make_view(params.material->base);
        _view_normal = make_view(params.material->normal);
        _view_height = make_view(params.material->height);
        _view_roughness = make_view(params.material->roughness);
        _view_ao = make_view(params.material->ao);

        /*
            Here we use spherical coordinates (θ,ϕ), where θ is the angle down from the
            pole, and ϕ is the angle around the axis through the pole. Now the hit point
            coordinates can be represented as
            x=cos(ϕ)cos(θ)
            y=sin(ϕ)cos(θ)
            z=sin(θ)
            u=ϕ/2π
            v=θ/π
        */

        // Determine bounding box
        bbox_min = glm::vec3(FLT_MAX, FLT_MAX, FLT_MAX);
        bbox_max = glm::vec3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
        for (auto &vtx : _mesh->position) {
            bbox_min = glm::min(bbox_min, vtx);
            bbox_max = glm::max(bbox_max, vtx);
        }

        auto bbox_extent = bbox_max - bbox_min;
        _radius = glm::max(bbox_extent.x, glm::max(bbox_extent.y, bbox_extent.z)) / 2;
        _center_of_mass = bbox_min + bbox_extent / 2.f;

        bbox_max = _center_of_mass + glm::vec3(1.5f * _radius, 1.5f * _radius, 1.5f * _radius);
        bbox_min = _center_of_mass - glm::vec3(1.5f * _radius, 1.5f * _radius, 1.5f * _radius);

        for (size_t i = 0; i < _alive_particle_limit; i++) {
            emit_new_particle();
        }
    }

    ~Old_Painter() override = default;

    void replace_particle(size_t idx) {
        auto dist_theta = std::uniform_real_distribution(0.f, glm::pi<float>());
        auto dist_phi = std::uniform_real_distribution(0.f, 2 * glm::pi<float>());

        auto theta = dist_theta(_rand);
        auto phi = dist_phi(_rand);

        auto x = glm::cos(phi) * glm::sin(theta);
        auto y = glm::sin(phi) * glm::sin(theta);
        auto z = glm::cos(theta);
        auto u = phi / (2 * glm::pi<float>());
        auto v = theta / glm::pi<float>();
        auto velocity = _radius / 16 * normalize(-glm::vec3(x, y, z));
        auto position = _center_of_mass + _radius * glm::vec3(x, y, z);
        auto uv = glm::vec2(u, v);

        auto color_base = sample_image(_view_base, uv);
        auto color_normal = sample_image(_view_normal, uv);
        auto color_height = sample_image(_view_height, uv);
        auto color_roughness = sample_image(_view_roughness, uv);
        auto color_ao = sample_image(_view_ao, uv);

        _particle_system.position[idx] = (position);
        _particle_system.next_position[idx] = (position);
        _particle_system.velocity[idx] = (velocity);
        _particle_system.color_base[idx] = (color_base);
        _particle_system.color_normal[idx] = (color_normal);
        _particle_system.color_roughness[idx] = (color_roughness);
        _particle_system.color_height[idx] = (color_height);
        _particle_system.color_ao[idx] = (color_ao);
        _particle_system.alive[idx] = true;

        _particles_to_be_emitted--;
    }

    void emit_new_particle() {
        auto idx = num_particles();

        _particle_system.position.push_back({});
        _particle_system.next_position.push_back({});
        _particle_system.velocity.push_back({});
        _particle_system.color_base.push_back({});
        _particle_system.color_normal.push_back({});
        _particle_system.color_roughness.push_back({});
        _particle_system.color_height.push_back({});
        _particle_system.color_ao.push_back({});
        _particle_system.alive.push_back(false);

        replace_particle(idx);
    }

    void step_painting(float dt) override {
        boost::thread_group threadpool;

        auto const N = num_particles();

        for (size_t pidx = 0; pidx < N; pidx++) {
            auto pos = _particle_system.next_position[pidx];
            if (!is_inside_bounding_box(bbox_min, bbox_max, pos)) {
                _particle_system.alive[pidx] = false;
                if (_particles_to_be_emitted > 0) {
                    replace_particle(pidx);
                }
            }
        }

        // Advance rays
        auto const worksize = 2048;

        auto func = [&](size_t index, size_t count) {
            for (size_t i = index; i < index + count; i++) {
                _particle_system.position[i] = _particle_system.next_position[i];
                _particle_system.next_position[i] = _particle_system.position[i] + dt * _particle_system.velocity[i];
            }
        };
        for (size_t i = 0; i < N; i += worksize) {
            size_t count = (i + worksize < N) ? worksize : N - i;
            size_t index = i;

            threadpool.create_thread(std::bind(func, index, count));
        }
        threadpool.join_all();

        // Generate intersections
        struct Intersection {
            size_t particle_idx;
            size_t triangle_idx;
            glm::vec3 xp;
            float s, t;
        };
        std::mutex lock_intersections;
        std::vector<Intersection> intersections;

        auto const num_triangles = _mesh->elements.size() / 3;
        auto func_check_intersect = [&](size_t i0, size_t i1) {
            for (size_t i = i0; i < i1; i++) {
                auto p0 = _particle_system.position[i];
                auto p1 = _particle_system.next_position[i];
                auto dir = p1 - p0;

                for (size_t triangle = 0; triangle < num_triangles; triangle++) {
                    auto element_offset = triangle * 3;
                    auto vi0 = _mesh->elements[element_offset + 0];
                    auto vi1 = _mesh->elements[element_offset + 1];
                    auto vi2 = _mesh->elements[element_offset + 2];
                    auto v0 = _mesh->position[vi0];
                    auto v1 = _mesh->position[vi1];
                    auto v2 = _mesh->position[vi2];
                    glm::vec3 xp;
                    float t;
                    if (intersect::ray_triangle(xp, t, p0, dir, v0, v1, v2)) {
                        if (0 <= t && t <= 1) {
                            auto surf_normal = cross(v1 - v0, v2 - v0);
                            bool hit_front = dot(dir, surf_normal) < 0.0f;
                            if (hit_front) {
                                auto coplanarity_score = coplanarity_test(v0, v1, v2, xp);
                                assert(fabs(coplanarity_score) < 0.01f);
                                float s, t;
                                if (parametric_form_of_point_on_triangle(v0, v1, v2, xp, s, t)) {
                                    std::lock_guard g(lock_intersections);
                                    intersections.emplace_back(Intersection{ i, triangle, xp, s, t });
                                }
                            }
                        }
                    }
                }
            }
        };
        for (size_t i = 0; i < N; i += worksize) {
            size_t count = (i + worksize < N) ? worksize : N - i;
            size_t index = i;

            threadpool.create_thread(std::bind(func_check_intersect, index, index + count));
        }
        threadpool.join_all();

        // Generate draw commands
        struct Draw_Command {
            glm::vec2 uv;
            Pixel3 base;
            Pixel3 normal;
            Pixel3 height;
            Pixel3 roughness;
            Pixel3 ao;
        };
        std::vector<Draw_Command> draw_commands;

        for (auto &intersection : intersections) {
            auto element_offset = intersection.triangle_idx * 3;
            auto vi0 = _mesh->elements[element_offset + 0];
            auto vi1 = _mesh->elements[element_offset + 1];
            auto vi2 = _mesh->elements[element_offset + 2];
            auto uv0 = _mesh->uv[vi0];
            auto uv1 = _mesh->uv[vi1];
            auto uv2 = _mesh->uv[vi2];

            auto uv = interpolate(uv0, uv1, uv2, intersection.s, intersection.t);
            assert(0 <= uv.x && uv.x <= 1);
            assert(0 <= uv.y && uv.y <= 1);
            auto pidx = intersection.particle_idx;
            draw_commands.emplace_back(
                Draw_Command{
                    uv,
                    _particle_system.color_base[pidx],
                    _particle_system.color_normal[pidx],
                    _particle_system.color_height[pidx],
                    _particle_system.color_roughness[pidx],
                    _particle_system.color_ao[pidx]
                }
            );
        }

        // Process draw commands
        auto view_out_base = make_view(_out_material, _out_material.base);
        for (auto &draw : draw_commands) {
            auto &pix = sample_image_ref(view_out_base, draw.uv);
            pix = draw.base;
            // TODO(danielm): rest of the textures
        }
    }

    bool is_painting_done() override {
        auto N = num_particles();

        size_t alive_particles = 0;
        for (size_t i = 0; i < N; i++) {
            if (_particle_system.alive[i]) {
                alive_particles++;
            }
        }

        auto particles_remain = _particles_to_be_emitted + alive_particles;
        printf("particles remaining: %zu\n", particles_remain);
        return particles_remain == 0;
    }

    size_t num_particles() override {
        return _particle_system.position.size();
    }

    bool get_particle(size_t id, glm::vec3 *out_position, glm::vec3 *out_next_position, glm::vec3 *out_velocity) override {
        if (id >= num_particles()) {
            return false;
        }

        *out_position = _particle_system.position[id];
        *out_next_position = _particle_system.next_position[id];
        *out_velocity = _particle_system.velocity[id];

        return true;
    }

    void result(PSP::Material *out_material) override {
        copy_texture_out(out_material->base, _out_material.dim, _out_material.base);
    }

    void copy_texture_out(PSP::Texture &dst, glm::ivec2 const &dim, std::unique_ptr<Pixel3[]> &src) {
        dst.width = dim.x;
        dst.height = dim.y;
        dst.buffer = src.get();
    }

private:
    PSP::Mesh const *_mesh;
    PSP::Material const *_in_material;
    size_t _particles_to_be_emitted;
    std::mt19937 _rand;

    float _radius;
    glm::vec3 _center_of_mass;

    Particle_System _particle_system;
    Out_Material _out_material;
    float _theta_step;
    float _phi_step;

    boost::gil::rgb8_view_t _view_base;
    boost::gil::rgb8_view_t _view_normal;
    boost::gil::rgb8_view_t _view_height;
    boost::gil::rgb8_view_t _view_roughness;
    boost::gil::rgb8_view_t _view_ao;

    glm::vec3 bbox_min, bbox_max;

    const size_t _alive_particle_limit = 2 * 16384;
};

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

            auto v = float(y) / float(_out_material.dim.y);
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
                        auto ray_normal = (n0 + n1 + n2) / 3.0f;

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

        printf("Raygen results: %zu\n", total_rays);
    }

    bool is_painting_done() override {
        return false;
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
