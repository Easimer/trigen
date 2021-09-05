// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#pragma once

#include <memory>

#include <softbody.h>

#if _WIN32
#if defined(FOLIAGE_BUILDING)
#define FOLIAGE_IMPORT __declspec(dllexport)
#else
#define FOLIAGE_IMPORT __declspec(dllimport)
#endif
#else
#define FOLIAGE_IMPORT 
#endif

enum class Foliage_Generator_Parameter_Name {
    EndOfList = 0,
    /**
     * Scale of the leaf quads; float.
     */
    Scale,
    /**
     * Seed used by the random number generators; unsigned.
     */
    Seed,
};

struct Foliage_Generator_Parameter {
    Foliage_Generator_Parameter_Name name;
    union {
        float f;
        unsigned u;
    } value;
};

class FOLIAGE_IMPORT IFoliage_Generator {
public:
    virtual ~IFoliage_Generator() = default;

    virtual bool
    generate()
        = 0;

    virtual uint32_t
    numVertices() const = 0;

    virtual uint32_t
    numElements() const = 0;

    virtual glm::vec3 const *
    positions() const = 0;

    virtual glm::vec3 const *
    normals() const = 0;

    virtual glm::vec2 const *
    texcoords() const = 0;

    virtual uint32_t const *
    elements() const = 0;
};

FOLIAGE_IMPORT
std::unique_ptr<IFoliage_Generator>
make_foliage_generator(sb::Unique_Ptr<sb::ISoftbody_Simulation> &simulation, Foliage_Generator_Parameter const *parameters);
