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
make_foliage_generator(sb::Unique_Ptr<sb::ISoftbody_Simulation> &simulation);
