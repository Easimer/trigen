// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#include <foliage.hpp>

#include <cstring>
#include <optional>
#include <random>
#include <unordered_set>

#include <trigen/mesh_compress.h>

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

static std::vector<glm::vec3>
collect_leaf_positions(Simulation_Ptr &simulation) {
    auto *plant = simulation->get_extension_plant_simulation();
    assert(plant != nullptr);

    if (plant == nullptr) {
        return {};
    }

    auto leaf_buds = plant->get_leaf_buds();
    auto leaf_bud_set
        = std::unordered_set<sb::index_t>(leaf_buds.cbegin(), leaf_buds.cend());

    std::vector<glm::vec3> leaf_positions;

    for (auto iter = simulation->get_particles(); !iter->ended();
         iter->step()) {
        auto particle = iter->get();
        if (leaf_bud_set.count(particle.id)) {
            leaf_positions.emplace_back(particle.position);
        }
    }

    return leaf_positions;
}

static std::vector<glm::quat>
generate_random_orientations(size_t N) {
    std::vector<glm::quat> ret;
    ret.reserve(N);

    // TODO(danielm): seed
    std::mt19937 rand(0);
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
    Buffer(TMC_Context ctx, std::vector<T> const &data) {
        TMC_CreateBuffer(ctx, &_handle, data.data(), data.size() * sizeof(T));
    }

    operator TMC_Buffer() const { return _handle; }

private:
    TMC_Buffer _handle;
};

template <typename T> struct Type;
template <> struct Type<float> { ETMC_Type type = k_ETMCType_Float32; };

template <typename T, unsigned N> class Attribute { public:
public:
    Attribute(TMC_Context ctx, TMC_Buffer buffer, TMC_Size stride, TMC_Size offset) {
        TMC_CreateAttribute(ctx, &_handle, buffer, N, Type<T>().type, stride, offset);
    }


    Attribute(TMC_Context ctx, TMC_Buffer buffer) : Attribute(ctx, buffer, N * sizeof(T), 0) {
    }

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
generate_quads(Foliage_Arrays const& foliage_arrays, float scale) {
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> texcoords;

    positions.reserve(foliage_arrays.positions.size() * 6);
    normals.reserve(foliage_arrays.positions.size() * 6);
    texcoords.reserve(foliage_arrays.positions.size() * 6);

    for (size_t i = 0; i < foliage_arrays.positions.size(); i++) {
        auto p = foliage_arrays.positions[i];
        auto R = foliage_arrays.orientations[i];
        auto up = R * glm::vec3(0, 1, 0)
            * conjugate(R);
        auto right = R * glm::vec3(1, 0, 0)
            * conjugate(R);
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
    Foliage_Generator(Simulation_Ptr &simulation, Foliage_Generator_Parameter const *parameters)
        : _simulation(simulation) {
        while (parameters != nullptr
               && parameters->name
                   != Foliage_Generator_Parameter_Name::EndOfList) {
            switch (parameters->name) {
            case Foliage_Generator_Parameter_Name::Scale: {
                _scale = parameters->value.f;
                assert(_scale > 0.0f);
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
        foliage.positions = collect_leaf_positions(_simulation);
        foliage.orientations
            = generate_random_orientations(foliage.positions.size());

        _mesh = generate_quads(foliage, _scale);

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
    float _scale = 1.0f;
};

FOLIAGE_IMPORT
std::unique_ptr<IFoliage_Generator>
make_foliage_generator(
    sb::Unique_Ptr<sb::ISoftbody_Simulation> &simulation,
    Foliage_Generator_Parameter const *parameters) {
    if (!simulation || !simulation->get_extension_plant_simulation()) {
        return nullptr;
    }

    return std::make_unique<Foliage_Generator>(simulation, parameters);
}
