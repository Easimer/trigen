// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: implementation 
//

#include "rt_intersect.h"

#include <cstdio>
#include <vector>
#include <list>

#include <cuda.h>
#include <cuda_runtime.h>

#include <optix.h>
#include <optix_host.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>

#include "optix_utils.h"
#include "params.h"

// Precompiled PTX of programs.cu
extern "C" char const *cuda_compile_ptx_1_generated_programs_cu_ptx;
extern "C" unsigned long long cuda_compile_ptx_1_generated_programs_cu_ptx_len;

#define OPTIX_KERNELS (cuda_compile_ptx_1_generated_programs_cu_ptx)
#define OPTIX_KERNELS_SIZ (cuda_compile_ptx_1_generated_programs_cu_ptx_len)

static void context_log_cb(
        unsigned level,
        char const *tag,
        char const *msg,
        void *user) {
    printf("[%u][%s] %s\n", level, tag, msg);
}

static optix::DeviceContext create_optix_context() {
    OptixDeviceContext context;
    CUcontext cuCtx = 0; // current ctx
    CHECK_OPTIX( optixInit() );
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &context_log_cb;
    options.logCallbackLevel = 4;
    CHECK_OPTIX( optixDeviceContextCreate(cuCtx, &options, &context) );

    return { context };
}

static optix::Module create_optix_module(OptixDeviceContext context, char const *ptx, unsigned long long ptx_siz, OptixPipelineCompileOptions const &pipeline_compile_options) {
    char log[2048];
    size_t sizeof_log = sizeof(log);

    OptixModuleCompileOptions module_compile_options = {};

    OptixModule module = nullptr;
    CHECK_OPTIX(
        optixModuleCreateFromPTX(
            context,
            &module_compile_options,
            &pipeline_compile_options,
            ptx,
            ptx_siz,
            log,
            &sizeof_log,
            &module
        )
    );

    return optix::Module(module);
}

static optix::ProgramGroup make_raygen_program_group(OptixDeviceContext context, OptixModule module) {
    char log[2048];
    size_t sizeof_log = sizeof(log);
    OptixProgramGroup prog_group = nullptr;
    OptixProgramGroupOptions program_group_options = {};
    OptixProgramGroupDesc prog_group_desc = {};

    prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    prog_group_desc.raygen.module = module;
    prog_group_desc.raygen.entryFunctionName = "__raygen__intersect";

    CHECK_OPTIX(
        optixProgramGroupCreate(
            context,
            &prog_group_desc,
            1, // num program groups
            &program_group_options,
            log,
            &sizeof_log,
            &prog_group
        )
    );

    return { prog_group };
}

static OptixProgramGroup make_miss_program_group(OptixDeviceContext context, OptixModule module) {
    char log[2048];
    size_t sizeof_log = sizeof(log);
    OptixProgramGroup prog_group = nullptr;
    OptixProgramGroupOptions program_group_options = {};
    OptixProgramGroupDesc prog_group_desc = {};

    prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    prog_group_desc.miss.module = module;
    prog_group_desc.miss.entryFunctionName = "__miss__intersect";

    CHECK_OPTIX(
        optixProgramGroupCreate(
            context,
            &prog_group_desc,
            1, // num program groups
            &program_group_options,
            log,
            &sizeof_log,
            &prog_group
        )
    );

    return prog_group;
}

static OptixProgramGroup make_hitgroup_program_group(OptixDeviceContext context, OptixModule module) {
    char log[2048];
    size_t sizeof_log = sizeof(log);
    OptixProgramGroup prog_group = nullptr;
    OptixProgramGroupOptions program_group_options = {};
    OptixProgramGroupDesc prog_group_desc = {};

    prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    prog_group_desc.hitgroup.moduleCH = module;
    prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__intersect";

    CHECK_OPTIX(
        optixProgramGroupCreate(
            context,
            &prog_group_desc,
            1, // num program groups
            &program_group_options,
            log,
            &sizeof_log,
            &prog_group
        )
    );

    return prog_group;
}

static optix::Pipeline make_pipeline(OptixDeviceContext context, OptixProgramGroup programs[3], OptixPipelineCompileOptions const &pipeline_compile_options) {
    char log[2048];
    size_t sizeof_log = sizeof(log);
    OptixPipeline pipeline = nullptr;
    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = 1;
    pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

    CHECK_OPTIX(optixPipelineCreate(
        context,
        &pipeline_compile_options,
        &pipeline_link_options,
        programs, 3,
        log,
        &sizeof_log,
        &pipeline
    ));

    return { pipeline };
}

template<typename T>
optix::SbtRecordHandle<T> make_record(T const &orig, OptixProgramGroup prog_grp) {
    optix::SbtRecord<T> record;
    auto const record_size = sizeof(record);
    void *ret = NULL;

    CHECK_CUDA( cudaMalloc(&ret, record_size) );

    record.data = orig;
    optixSbtRecordPackHeader(prog_grp, &record);

    CHECK_CUDA( cudaMemcpy(ret, &record, record_size, cudaMemcpyHostToDevice) );

    return { ret };
}

static bool load_ptx(std::vector<uint8_t> *ptx, char const *path) {
    FILE *f;
    long fileSiz;

    f = fopen(path, "rb");
    if(f == NULL) {
        return false;
    }

    fseek(f, 0, SEEK_END);
    fileSiz = ftell(f);
    fseek(f, 0, SEEK_SET);

    ptx->clear();
    ptx->resize(fileSiz + 1);
    fread(ptx->data(), fileSiz, 1, f);
    ptx->data()[fileSiz] = '\0';

    fclose(f);
    return true;
}

namespace rt_intersect {
    template<typename RaygenRecord, typename HitgroupRecord, typename MissRecord>
    class SBT {
    public:
        SBT(optix::PipelineManager& pmgr, RaygenRecord const &recRg, HitgroupRecord const &recHit, MissRecord const &recMiss) :
            _sbt{},
            _d_sbtRaygen(make_record(recRg, pmgr._pg_raygen)),
            _d_sbtClosestHit(make_record(recHit, pmgr._pg_hit)),
            _d_sbtMiss(make_record(recMiss, pmgr._pg_miss)) {

            _sbt.raygenRecord = _d_sbtRaygen;

            _sbt.hitgroupRecordBase = _d_sbtClosestHit;
            _sbt.hitgroupRecordStrideInBytes = _d_sbtClosestHit.stride();
            _sbt.hitgroupRecordCount = 1;

            _sbt.missRecordBase = _d_sbtMiss;
            _sbt.missRecordStrideInBytes = _d_sbtMiss.stride();
            _sbt.missRecordCount = 1;
        }

        SBT(SBT &&other) : _sbt{}, _d_sbtRaygen{}, _d_sbtClosestHit{}, _d_sbtMiss{} {
            std::swap(_sbt, other._sbt);
            std::swap(_d_sbtRaygen, other._d_sbtRaygen);
            std::swap(_d_sbtClosestHit, other._d_sbtClosestHit);
            std::swap(_d_sbtMiss, other._d_sbtMiss);
        }

        OptixShaderBindingTable _sbt;
        optix::SbtRecordHandle<RaygenRecord> _d_sbtRaygen;
        optix::SbtRecordHandle<HitgroupRecord> _d_sbtClosestHit;
        optix::SbtRecordHandle<MissRecord> _d_sbtMiss;
    };

    template<typename T>
    T *cudaMallocThrows(size_t count) {
        T *ret = nullptr;
        auto rc = cudaMalloc(&ret, count * sizeof(T));

        if (rc != cudaSuccess) {
            throw std::exception("cudaMalloc failed");
        }

        return ret;
    }

    class Mesh_Attributes {
    public:
        Mesh_Attributes(Mesh_Descriptor const *mesh)
            : _d_normals(nullptr), _d_normal_indices(nullptr) {
            _d_normals = cudaMallocThrows<float3>(mesh->num_normals);
            _d_normal_indices = cudaMallocThrows<unsigned>(mesh->num_triangles * 3);
            CHECK_CUDA(cudaMemcpy(_d_normals, mesh->h_normals, mesh->num_normals * sizeof(float3), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(_d_normal_indices, mesh->h_normal_indices, mesh->num_triangles * sizeof(uint3), cudaMemcpyHostToDevice));
        }

        ~Mesh_Attributes() {
            cudaFree(_d_normal_indices);
            cudaFree(_d_normals);
        }

        float3 *normals() { return _d_normals; }
        unsigned *normal_indices() { return _d_normal_indices; }
    private:
        float3 *_d_normals;
        unsigned *_d_normal_indices;
    };
    
    class Instance_ID_Allocator {
    public:
        unsigned allocate() {
            auto current = _ids.begin();
            auto end = _ids.end();

            // No ID in the list yet or the first ID is not zero
            if (_ids.empty() || *current != 0) {
                _ids.push_front(0);
                return 0;
            }

            // Find a pair of neighboring nodes in the list such that
            // there is a gap between their values
            // e.g. if there are two nodes with values 0 and 2, that means that
            // there was an ID of 1, but it was deallocated and we can reuse it
            auto next = ++(_ids.begin());

            while (next != end) {
                if (*next != *current + 1) {
                    // There is a gap between the values of current and next
                    auto ret = *current + 1;
                    // Insert new ID after current and return
                    _ids.insert(std::next(current), ret);
                    return ret;
                }

                ++current;
                ++next;
            }

            auto ret = *current + 1;
            _ids.insert(next, ret);

            return ret;
        }

        void deallocate(unsigned id) {
            // Find ID in list
            auto it = std::find(_ids.begin(), _ids.end(), id);
            // Check in debug builds if the caller passed us a bad ID
            assert(it != _ids.end());
            // Remove ID
            _ids.erase(it);
        }

        unsigned size() const {
            if (_ids.empty()) {
                return 0;
            }

            return _ids.back() + 1;
        }

    private:
        std::list<unsigned> _ids;
    };

    class Mesh : public IMesh {
    public:
        Mesh(optix::DeviceContext &ctx, Instance_ID_Allocator *id_allocator, Mesh_Descriptor const *mesh, float const *transform) :
            _gas(ctx, mesh->num_triangles, mesh->num_vertices, mesh->h_vertices, mesh->h_vertex_indices),
            _id_allocator(id_allocator),
            _id(id_allocator->allocate()),
            _attr(mesh) {
            set_transform(transform);
        }

        ~Mesh() {
            _id_allocator->deallocate(_id);
        }

        OptixTraversableHandle traversable_handle() { return _gas.handle(); }
        float3 *normals() { return _attr.normals(); }
        unsigned *normal_indices() { return _attr.normal_indices(); }

        float const *transform() const { return _transform; }
        void set_transform(float const *transform) override {
            // transform is assumed to be a 4x4 column-major matrix (glm::mat4)
            // but optix wants a 3x4 row-major matrix
            for (int col = 0; col < 4; col++) {
                for (int row = 0; row < 3; row++) {
                    auto dst_idx = row * 4 + col;
                    auto src_idx = col * 4 + row;
                    _transform[dst_idx] = transform[src_idx];
                }
            }
        }

        unsigned id() const { return _id; }

    private:
        optix::GAS _gas;
        Instance_ID_Allocator *_id_allocator;
        unsigned _id;
        Mesh_Attributes _attr;
        float _transform[12];
    };

    Mesh **null_mesh_iterator = NULL;

    class RTInstance : public IInstance {
    public:
        RTInstance(
            optix::Library &&library,
            optix::DeviceContext &&context,
            SBT<Sbt_Raygen, Sbt_ClosestHit, Sbt_Miss> &&sbt,
            optix::PipelineManager &&pipeline
        ) :
            _library(std::move(library)),
            _context(std::move(context)),
            _sbt(std::move(sbt)),
            _pipeline(std::move(pipeline)),
            _ias(optix::make_instance_AS(_context, null_mesh_iterator, null_mesh_iterator)) {
        }

        Shared_Ptr<IMesh> upload_mesh(Mesh_Descriptor const *mesh) override {
            if (mesh->num_triangles == 0 ||
                mesh->num_vertices == 0 ||
                mesh->num_normals == 0) {
                return nullptr;
            }
            if (mesh->h_vertices == nullptr ||
                mesh->h_vertex_indices == nullptr ||
                mesh->h_normals == nullptr ||
                mesh->h_normal_indices == nullptr ||
                mesh->transform == nullptr) {
                return nullptr;
            }

            auto ret = std::make_shared<Mesh>(_context, &_iid_allocator, mesh, mesh->transform);
            _meshes.push_back(std::weak_ptr(ret));

            invalidate_instance_AS();

            return ret;
        }

        Status exec(Ray_Bundle const *rays, Results *results, CUstream stream) override {
            if (rays == nullptr || results == nullptr) {
                return ESTATUS_ARGS;
            }
            if (rays->d_origins == nullptr ||
                rays->d_directions == nullptr) {
                return ESTATUS_ARGS;
            }
            if (results->d_ray_index == nullptr ||
                results->d_flags == nullptr ||
                results->d_xp == nullptr ||
                results->d_surf_normal == nullptr ||
                results->d_depth == nullptr
                ) {
                return ESTATUS_ARGS;
            }
            if (results->num_results < rays->num_rays) {
                return ESTATUS_NOT_ENOUGH_SPACE;
            }

            optix::Constant<Params> params;

            // TODO: check args

            if (instance_AS_needs_rebuild()) {
                invalidate_instance_AS();
            }

            // Take temp ownership of all meshes by transforming all weak ptr's
            // into shared ones 
            std::vector<std::shared_ptr<Mesh>> owned_meshes;
            std::transform(
                _meshes.begin(), _meshes.end(),
                std::back_inserter(owned_meshes),
                [&](std::weak_ptr<Mesh> &mesh_weak_ptr) -> std::shared_ptr<Mesh> {
                    return mesh_weak_ptr.lock();
                }
            );

            auto size = _iid_allocator.size();
            std::vector<float3 *> h_normals;
            std::vector<unsigned *> h_normal_indices;

            h_normals.resize(size);
            h_normal_indices.resize(size);

            for (auto &mesh : owned_meshes) {
                if (mesh != nullptr) {
                    auto id = mesh->id();
                    h_normals[id] = mesh->normals();
                    h_normal_indices[id] = mesh->normal_indices();
                }
            }

            // TODO(danielm): use CUDA_Array when we move this code into softbody
            CHECK_CUDA(cudaMalloc(&params->normals, size * sizeof(float3 *)));
            CHECK_CUDA(cudaMalloc(&params->normal_indices, size * sizeof(unsigned *)));

            CHECK_CUDA(cudaMemcpy(params->normals, h_normals.data(), size * sizeof(float3 *), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(params->normal_indices, h_normal_indices.data(), size * sizeof(unsigned *), cudaMemcpyHostToDevice));

            params->ray_origins = rays->d_origins;
            params->ray_directions = rays->d_directions;

            params->ray_index = results->d_ray_index;
            params->flags = results->d_flags;
            params->xp = results->d_xp;
            params->surf_normal = results->d_surf_normal;
            params->depth = results->d_depth;
            
            params->handle = _ias.traversable_handle();

            params.upload();

            auto rc = optixLaunch(
                _pipeline._pipeline,
                stream,
                (CUdeviceptr)params.d_ptr,
                sizeof(params.h_data),
                &_sbt._sbt,
                rays->num_rays,
                /* height: */ 1,
                /* depth: */ 1);

            cudaFree(params->normal_indices);
            cudaFree(params->normals);

            return ESTATUS_OK;
        }

        bool instance_AS_needs_rebuild() {
            return std::any_of(
                _meshes.begin(), _meshes.end(),
                [&](auto ptr) { return ptr.expired(); }
            );
        }

        void invalidate_instance_AS() {
            // Remove expired meshes from our list
            std::remove_if(
                _meshes.begin(), _meshes.end(),
                [&](std::weak_ptr<Mesh> &mesh_ptr) {
                    return mesh_ptr.expired();
                }
            );

            // Take temp ownership of all meshes by transforming all weak ptr's
            // into shared ones 
            std::vector<std::shared_ptr<Mesh>> owned_meshes;
            std::transform(
                _meshes.begin(), _meshes.end(),
                std::back_inserter(owned_meshes),
                [&](std::weak_ptr<Mesh> &mesh_weak_ptr) -> std::shared_ptr<Mesh> {
                    return mesh_weak_ptr.lock();
                }
            );

            _ias = optix::make_instance_AS(_context, owned_meshes.begin(), owned_meshes.end());
        }

        void notify_meshes_updated() override {
            invalidate_instance_AS();
        }

    private:
        optix::Library _library;
        optix::DeviceContext _context;
        SBT<Sbt_Raygen, Sbt_ClosestHit, Sbt_Miss> _sbt;
        optix::PipelineManager _pipeline;
        std::list<std::weak_ptr<Mesh>> _meshes;
        optix::Instance_AS _ias;
        Instance_ID_Allocator _iid_allocator;
    };


    Unique_Ptr<IInstance> make_instance() {
        auto library = optix::Library();
        auto context = create_optix_context();

        // Pipeline compile options 
        OptixPipelineCompileOptions pipeline_compile_options = {};
        pipeline_compile_options.usesMotionBlur = false;

        pipeline_compile_options.traversableGraphFlags =
            OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
        pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
        pipeline_compile_options.numPayloadValues = 8;

        // Loading our module
        auto module = create_optix_module(context, OPTIX_KERNELS, OPTIX_KERNELS_SIZ, pipeline_compile_options);
        auto raygen_pgrp = make_raygen_program_group(context, module);
        auto miss_pgrp = make_miss_program_group(context, module);
        auto hit_pgrp = make_hitgroup_program_group(context, module);
        OptixProgramGroup programs[] = { raygen_pgrp, miss_pgrp, hit_pgrp };

        auto pipeline = make_pipeline(context, programs, pipeline_compile_options);

        optix::PipelineManager pipeline_mgr(
            std::move(pipeline),
            std::move(raygen_pgrp),
            std::move(hit_pgrp),
            std::move(miss_pgrp),
            std::move(module)
        );

        // Making SBT records
        Sbt_Raygen h_sbtRaygen = {};
        Sbt_ClosestHit h_sbtClosestHit = {};
        Sbt_Miss h_sbtMiss = {};

        SBT sbt(pipeline_mgr, h_sbtRaygen, h_sbtClosestHit, h_sbtMiss);

        // TODO: move above things into ctor of RTInstance
        return std::make_unique<RTInstance>(
            std::move(library),
            std::move(context),
            std::move(sbt),
            std::move(pipeline_mgr)
        );
    }
}