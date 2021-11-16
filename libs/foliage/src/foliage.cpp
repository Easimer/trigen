// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#include <foliage.hpp>

#include <cstring>
#include <mutex>
#include <optional>
#include <random>
#include <unordered_set>

#include <trigen/mesh_compress.h>
#include <worker_group.hpp>

using Simulation_Ptr = sb::Unique_Ptr<sb::ISoftbody_Simulation>;

struct Foliage_Arrays {
    std::vector<glm::vec3> positions;
    std::vector<glm::quat> orientations;
};

struct Foliage_Mesh_Arrays {
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> texcoords;
    std::vector<uint32_t> elements;
};

struct Foliage_Params {
    float scale = 1.0f;
    float radius = 1.0f;
    float density = 1.0f;
    unsigned rnd_seed = 0;
};

static float
distance_to_closest_leaf_bud(
    std::vector<glm::vec3> const &leaf_positions,
    glm::vec3 const &cur,
    float foliage_radius) {
    float min_dist = foliage_radius;
    // Compute the minimum of the distances between `cur`
    // and the leaves
    for (auto const &leaf_bud_position : leaf_positions) {
        const auto dist
            = glm::min(distance(cur, leaf_bud_position), foliage_radius);
        min_dist = glm::min(min_dist, dist);

        if (glm::epsilonEqual(min_dist, 0.0f, 0.001f)) {
            break;
        }
    }
    return min_dist;
}

static std::vector<glm::vec3>
collect_leaf_positions(
    Foliage_Params const &params,
    Simulation_Ptr &simulation) {
    auto *plant = simulation->get_extension_plant_simulation();
    assert(plant != nullptr);

    if (plant == nullptr) {
        return {};
    }

    auto leaf_buds = plant->get_leaf_buds();
    auto apical_children = plant->get_apical_children();
    auto lateral_buds = plant->get_lateral_buds();
    std::unordered_set<sb::index_t> leaf_bud_set;
    for (auto const idx : leaf_buds)
        leaf_bud_set.insert(idx);
    for (auto const idx : apical_children)
        leaf_bud_set.insert(idx);

    float const foliage_radius = params.radius;
    float const foliage_density = params.density;
    glm::vec3 leaf_min(+INFINITY, +INFINITY, +INFINITY);
    glm::vec3 leaf_max(-INFINITY, -INFINITY, -INFINITY);

    std::vector<glm::vec3> leaf_positions;
    leaf_positions.reserve(leaf_bud_set.size());

    for (auto iter = simulation->get_particles(); !iter->ended();
         iter->step()) {
        auto particle = iter->get();
        if (leaf_bud_set.count(particle.id) != 0) {
            leaf_min = glm::min(leaf_min, particle.position);
            leaf_max = glm::max(leaf_max, particle.position);
            leaf_positions.emplace_back(particle.position);
        }
    }

    assert(leaf_positions.size() == leaf_bud_set.size());

    leaf_min -= glm::vec3(foliage_radius, foliage_radius, foliage_radius);
    leaf_max += glm::vec3(foliage_radius, foliage_radius, foliage_radius);
    auto step = (1 / 8.0f) * (leaf_max - leaf_min);

    Worker_Group workers;

    std::mutex leaves_lock;
    std::vector<glm::vec3> leaves;

    auto task_leaf_gen = [&](float x0, float x1, float y0, float y1, float z0,
                             float z1) {
        std::mt19937 rand(params.rnd_seed);
        std::uniform_real_distribution dist(0.f, 1.f);
        std::vector<glm::vec3> local_leaves;

        const auto step = glm::max(1 / foliage_density, 0.001f);
        assert(step > glm::epsilon<float>());

        for (float x = x0; x < x1; x += step) {
            for (float y = y0; y < y1; y += step) {
                for (float z = z0; z < z1; z += step) {
                    glm::vec3 cur = { x, y, z };

                    auto min_dist = distance_to_closest_leaf_bud(
                        leaf_positions, cur, foliage_radius);

                    if (min_dist > foliage_radius) {
                        continue;
                    }

                    const auto dice_roll = dist(rand);
                    const auto prob = 1 - glm::sqrt(min_dist / foliage_radius);
                    if (prob >= dice_roll) {
                        local_leaves.push_back(cur);
                    }
                }
            }
        }

        std::lock_guard G(leaves_lock);
        leaves.insert(leaves.end(), local_leaves.cbegin(), local_leaves.cend());
    };

    struct Range {
        float x0, x1, y0, y1, z0, z1;
    };
    std::vector<Range> ranges;
    for (float x0 = leaf_min.x; x0 < leaf_max.x; x0 += step.x) {
        float x1 = x0 + step.x;
        assert(x0 < x1);
        for (float y0 = leaf_min.y; y0 < leaf_max.y; y0 += step.y) {
            float y1 = y0 + step.y;
            assert(y0 < y1);
            for (float z0 = leaf_min.z; z0 < leaf_max.z; z0 += step.z) {
                float z1 = z0 + step.z;
                assert(z0 < z1);
                ranges.push_back({ x0, x1, y0, y1, z0, z1 });
            }
        }
    }

    for (auto &range : ranges) {
        workers.emplace_task([&, range]() {
            task_leaf_gen(
                range.x0, range.x1, range.y0, range.y1, range.z0, range.z1);
        });
    }
    workers.wait();

    return leaves;
}

static std::vector<glm::quat>
generate_random_orientations(Foliage_Params const &params, size_t N) {
    std::vector<glm::quat> ret;
    ret.reserve(N);

    std::mt19937 rand(params.rnd_seed);
    std::uniform_real_distribution dist(-1.0f, 1.0f);

    while (N > 0) {
        glm::quat q(dist(rand), dist(rand), dist(rand), dist(rand));
        ret.emplace_back(glm::normalize(q));
        N--;
    }

    return ret;
}

namespace tmc {
template <typename T> class Buffer {
public:
    Buffer(TMC_Context ctx, std::vector<T> const &data)
        : _handle(nullptr) {
        TMC_CreateBuffer(ctx, &_handle, data.data(), data.size() * sizeof(T));
    }

    operator TMC_Buffer() const { return _handle; }

private:
    TMC_Buffer _handle;
};

template <typename T> struct Type;
template <> struct Type<float> { ETMC_Type type = k_ETMCType_Float32; };

template <typename T, unsigned N> class Attribute {
public:
    Attribute(
        TMC_Context ctx,
        TMC_Buffer buffer,
        TMC_Size stride,
        TMC_Size offset)
        : _handle(nullptr) {
        TMC_CreateAttribute(
            ctx, &_handle, buffer, N, Type<T>().type, stride, offset);
    }

    Attribute(TMC_Context ctx, TMC_Buffer buffer)
        : Attribute(ctx, buffer, N * sizeof(T), 0) { }

    operator TMC_Attribute() const { return _handle; }

private:
    TMC_Attribute _handle;
};
}

template <typename T, unsigned N>
static std::vector<glm::vec<N, T>>
to_glm_vector(TMC_Context ctx, tmc::Attribute<T, N> &attr) {
    void const *data;
    TMC_Size size;

    TMC_GetDirectArray(ctx, attr, &data, &size);

    std::vector<glm::vec<N, T>> ret;

    auto count = size / sizeof(glm::vec<N, T>);
    ret.resize(count);

    memcpy(ret.data(), data, size);

    return ret;
}

static void
compress_mesh(
    Foliage_Mesh_Arrays &out,
    std::vector<glm::vec3> const &positions,
    std::vector<glm::vec3> const &normals,
    std::vector<glm::vec2> const &texcoords) {
    TMC_Context ctx;

    auto num_vertices = positions.size();

    TMC_CreateContext(&ctx, k_ETMCHint_None);
    TMC_SetIndexArrayType(ctx, k_ETMCType_UInt32);
    // Mesh is mostly quads
    TMC_SetParamInteger(ctx, k_ETMCParam_WindowSize, 4);

    tmc::Buffer bufPos(ctx, positions);
    tmc::Buffer bufNormal(ctx, normals);
    tmc::Buffer bufTexcoord(ctx, texcoords);

    tmc::Attribute<float, 3> attrPos(ctx, bufPos);
    tmc::Attribute<float, 3> attrNormal(ctx, bufNormal);
    tmc::Attribute<float, 2> attrTexcoord(ctx, bufTexcoord);

    TMC_Compress(ctx, num_vertices);

    out.positions = to_glm_vector(ctx, attrPos);
    out.normals = to_glm_vector(ctx, attrNormal);
    out.texcoords = to_glm_vector(ctx, attrTexcoord);

    void const *indices;
    TMC_Size indices_size;
    TMC_Size indices_count;
    TMC_GetIndexArray(ctx, &indices, &indices_size, &indices_count);
    out.elements.resize(indices_count);
    memcpy(out.elements.data(), indices, indices_size);

    TMC_DestroyContext(ctx);
}

static Foliage_Mesh_Arrays
generate_quads(
    Foliage_Params const &params,
    Foliage_Arrays const &foliage_arrays) {
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> texcoords;

    positions.reserve(foliage_arrays.positions.size() * 6);
    normals.reserve(foliage_arrays.positions.size() * 6);
    texcoords.reserve(foliage_arrays.positions.size() * 6);

    auto const scale = params.scale;

    for (size_t i = 0; i < foliage_arrays.positions.size(); i++) {
        auto p = foliage_arrays.positions[i];
        auto R = foliage_arrays.orientations[i];
        auto up = R * glm::vec3(0, 1, 0) * conjugate(R);
        auto right = R * glm::vec3(1, 0, 0) * conjugate(R);
        auto normal = cross(right, up);

        up *= scale;
        right *= scale;

        auto v0 = p + 0.5f * up - 0.5f * right;
        auto v1 = p - 0.5f * up - 0.5f * right;
        auto v2 = p - 0.5f * up + 0.5f * right;
        auto v3 = p + 0.5f * up + 0.5f * right;

        positions.emplace_back(v0);
        positions.emplace_back(v1);
        positions.emplace_back(v2);
        positions.emplace_back(v0);
        positions.emplace_back(v2);
        positions.emplace_back(v3);

        texcoords.emplace_back(0, 1);
        texcoords.emplace_back(0, 0);
        texcoords.emplace_back(1, 0);
        texcoords.emplace_back(0, 1);
        texcoords.emplace_back(1, 0);
        texcoords.emplace_back(1, 1);

        for (int j = 0; j < 6; j++) {
            normals.push_back(normal);
        }
    }

    Foliage_Mesh_Arrays ret;
    compress_mesh(ret, positions, normals, texcoords);

    return ret;
}

class Foliage_Generator : public IFoliage_Generator {
public:
    ~Foliage_Generator() override = default;
    Foliage_Generator(
        Simulation_Ptr &simulation,
        Foliage_Generator_Parameter const *parameters)
        : _simulation(simulation) {
        while (parameters != nullptr
               && parameters->name
                   != Foliage_Generator_Parameter_Name::EndOfList) {
            switch (parameters->name) {
            case Foliage_Generator_Parameter_Name::Scale: {
                _params.scale = parameters->value.f;
                assert(_params.scale > 0.0f);
                break;
            }
            case Foliage_Generator_Parameter_Name::Radius: {
                _params.radius = parameters->value.f;
                assert(_params.radius > 0.0f);
                break;
            }
            case Foliage_Generator_Parameter_Name::Seed: {
                _params.rnd_seed = parameters->value.u;
                break;
            }
            case Foliage_Generator_Parameter_Name::Density: {
                _params.density = parameters->value.f;
                break;
            }
            default: {
                assert(!"Unhandled parameter");
                break;
            }
            }

            parameters++;
        }
    }

    bool
    generate() override {
        Foliage_Arrays foliage;
        foliage.positions = collect_leaf_positions(_params, _simulation);

        if (foliage.positions.empty()) {
            return false;
        }

        foliage.orientations
            = generate_random_orientations(_params, foliage.positions.size());

        _mesh = generate_quads(_params, foliage);

        return true;
    }

    uint32_t
    numVertices() const override {
        if (!_mesh)
            return 0;
        return _mesh->positions.size();
    }

    uint32_t
    numElements() const override {
        if (!_mesh)
            return 0;
        return _mesh->elements.size();
    }

    glm::vec3 const *
    positions() const override {
        if (!_mesh)
            return nullptr;
        return _mesh->positions.data();
    }

    glm::vec3 const *
    normals() const override {
        if (!_mesh)
            return nullptr;
        return _mesh->normals.data();
    }

    glm::vec2 const *
    texcoords() const override {
        if (!_mesh)
            return nullptr;
        return _mesh->texcoords.data();
    }

    uint32_t const *
    elements() const override {
        if (!_mesh)
            return nullptr;
        return _mesh->elements.data();
    }

private:
    Simulation_Ptr &_simulation;
    std::optional<Foliage_Mesh_Arrays> _mesh;
    Foliage_Params _params;
};

FOLIAGE_IMPORT
std::unique_ptr<IFoliage_Generator>
make_foliage_generator(
    sb::Unique_Ptr<sb::ISoftbody_Simulation> &simulation,
    Foliage_Generator_Parameter const *parameters) {
    if (simulation == nullptr
        || !simulation->get_extension_plant_simulation()) {
        return nullptr;
    }

    return std::make_unique<Foliage_Generator>(simulation, parameters);
}
