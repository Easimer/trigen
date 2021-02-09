// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: rt_intersect API
//

#pragma once

#include <memory>

#include <cuda.h>
#include <cuda_runtime.h>

namespace rt_intersect {
    template<typename T> using Unique_Ptr = std::unique_ptr<T>;
    template<typename T> using Shared_Ptr = std::shared_ptr<T>;

    enum Trace_Flags {
        ETRACE_NONE     = 0,
        ETRACE_HIT      = 1 << 0,

        ETRACE_MAX      = 1 << 7
    };

    enum Status {
        ESTATUS_OK = 0,
        ESTATUS_ARGS = 1,
        ESTATUS_NOT_ENOUGH_SPACE = 2,
        ESTATUS_MAX
    };

    class IMesh {
    public:
        virtual ~IMesh() = default;

        /*
         * Set the transformation matrix for this object.
         *
         * @param transform Pointer to a 4x4 column-major transformation matrix.
         *
         * @note Call notify_meshes_updated() on your IInstance after you call this.
         */
        virtual void set_transform(float const *transform) = 0;
    };

    struct Mesh_Descriptor {
        // Pointer to a 4x4 column-major transform matrix
        float *transform;

        // Number of triangles
        unsigned num_triangles;
        // Array of vertex indices; 3 indices for each triangle
        unsigned *h_vertex_indices;
        // Array of normal indices; 3 indices for each triangle
        unsigned *h_normal_indices;

        // Number of vertices
        unsigned num_vertices;
        // Array of vertex positions; must be atleast num_vertices long
        float3 *h_vertices;
        // Number of vertex normals
        unsigned num_normals;
        // Array of vertex normals; must be atleast num_normals long
        float3 *h_normals;
    };

    struct Ray_Bundle {
        // Number of rays
        unsigned num_rays;
        // Array of ray origins; must be atleast num_rays long
        float3 *d_origins;
        // Array of ray directions; must be atleast num_rays long
        float3 *d_directions;
    };

    struct Results {
        // Number of results, that is, the length of each array below.
        // When you pass this struct into the exec() method, this field tells
        // the algorithm how much space is available.
        // When the method has returned, this field may contain a value less
        // than originally. The algorithm may sort the results by the d_flags array,
        // so that rays that have hit are at the beginning, and those that have missed
        // will be at the end.
        //
        // E.g. if you have two rays and only one of them hits a surface, then
        // when exec() returns the number of results may be only one instead of
        // two.
        //
        // Nevertheless you should still check the flags because this sorting
        // mechanism is not yet implemented.
        unsigned num_results;

        // Ray index buffer
        unsigned *d_ray_index;

        // An array of bitfields of `enum Trace_Flags`
        // If bit ETRACE_HIT is set then the ray has hit a triangle
        uint8_t *d_flags;

        // An array of intersection points
        float3 *d_xp;
        // An array of surface normals at the intersection point
        float3 *d_surf_normal;
        // An array of penetration depths
        float *d_depth;
    };

    class IInstance {
    public:
        virtual ~IInstance() = default;

        /*
         *
         * @note this method, the destructor of IMesh and the exec method
         * below are all thread-unsafe.
         * Make sure that you arent destroy meshes on one thread and creating
         * new ones on an other thread.
         */
        virtual Shared_Ptr<IMesh> upload_mesh(Mesh_Descriptor const *mesh) = 0;

        /*
         * @note the arrays in the results struct must have enough space to
         * the results of all arrays;
         * that is rays->num_rays == results->num_results.
         */
        virtual Status exec(Ray_Bundle const *rays, Results *results, CUstream stream) = 0;

        /*
         * Tell this instance that one or more meshes have been updated.
         * This is an expensive operation and it's recommended that you do all
         * your mesh updating in a batch and only then call this method.
         */
        virtual void notify_meshes_updated() = 0;
    };

    Unique_Ptr<IInstance> make_instance();
}