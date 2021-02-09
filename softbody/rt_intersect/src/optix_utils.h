// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: resource manager classes for OptiX stuff
//

#pragma once

#include <cassert>

#include <cuda_runtime.h>

#include <optix.h>
#include <optix_host.h>
#include <optix_stubs.h>
#include <optix_function_table.h>

#define CHECK_CUDA(expr) \
{                                       \
    cudaError _RES_;                    \
    _RES_ = expr;                       \
    if(_RES_ != CUDA_SUCCESS) {         \
        throw optix::exception(_RES_);  \
    }                                   \
}

#define CHECK_OPTIX(expr) \
{                                       \
    OptixResult _RES_;                  \
    _RES_ = expr;                       \
    if(_RES_ != OPTIX_SUCCESS) {        \
        throw optix::exception(_RES_);  \
    }                                   \
}

namespace optix {
    template<typename ResultCodeT>
    class exception : public std::exception {
    public:
        exception(ResultCodeT result) :
            _rc(result) {
        }

        ResultCodeT result_code() const {
            return _rc;
        }

    private:
        ResultCodeT _rc;
    };

    struct Library {
        Library() {
            optixInit();
        }
    };

    template <typename T>
    struct SbtRecord {
        __align__(OPTIX_SBT_RECORD_ALIGNMENT)
            char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        T data;
    };

    template<typename T>
    struct SbtRecordHandle {
        SbtRecordHandle() : _d_ptr(nullptr) {}
        SbtRecordHandle(void *d_ptr) : _d_ptr(d_ptr) {}

        SbtRecordHandle(SbtRecordHandle<T> const &) = delete;
        void operator=(SbtRecordHandle<T> const &) = delete;

        ~SbtRecordHandle() {
            cudaFree(_d_ptr);
        }

        SbtRecordHandle(SbtRecordHandle<T> &&other) : _d_ptr(nullptr) {
            std::swap(_d_ptr, other._d_ptr);
        }

        SbtRecordHandle &operator=(SbtRecordHandle<T> &&other) {
            cudaFree(_d_ptr);
            _d_ptr = nullptr;
            std::swap(_d_ptr, other._d_ptr);
            return *this;
        }

        operator void *() const { return _d_ptr; }
        operator CUdeviceptr() const { return (CUdeviceptr)_d_ptr; }

        constexpr unsigned stride() const { return sizeof(SbtRecord<T>); }
    private:
        void *_d_ptr;
    };

    template<typename T>
    struct Constant {
        T h_data;
        void *d_ptr;

        Constant() : h_data{}, d_ptr(NULL) {
            CHECK_CUDA(cudaMalloc(&d_ptr, sizeof(T)));
        }

        ~Constant() {
            cudaFree(d_ptr);
        }

        T *operator->() {
            return &h_data;
        }

        void upload() {
            cudaMemcpy(d_ptr, &h_data, sizeof(T), cudaMemcpyHostToDevice);
        }
    };

    template<typename Handle, typename Destroyer>
    class OptixHandle {
    public:
        OptixHandle() : _handle(nullptr) {}

        OptixHandle(OptixHandle &&other) : _handle(nullptr) {
            std::swap(_handle, other._handle);
        }

        OptixHandle(Handle handle) : _handle(handle) {}

        ~OptixHandle() {
            if (_handle != nullptr) {
                Destroyer d;
                d(_handle);
            }
        }

        operator Handle() {
            return _handle;
        }

        OptixHandle &operator=(OptixHandle &&other) {
            std::swap(_handle, other._handle);
            return *this;
        }

    private:
        Handle _handle;
    };

    struct DeviceContextDestroyer {
        void operator()(OptixDeviceContext handle) {
            CHECK_OPTIX(optixDeviceContextDestroy(handle));
        }
    };

    struct ModuleDestroyer {
        void operator()(OptixModule handle) {
            CHECK_OPTIX(optixModuleDestroy(handle));
        }
    };

    struct PipelineDestroyer {
        void operator()(OptixPipeline handle) {
            CHECK_OPTIX(optixPipelineDestroy(handle));
        }
    };

    struct ProgramGroupDestroyer {
        void operator()(OptixProgramGroup handle) {
            CHECK_OPTIX(optixProgramGroupDestroy(handle));
        }
    };

    using DeviceContext = OptixHandle<OptixDeviceContext, DeviceContextDestroyer>;
    using Module = OptixHandle<OptixModule, ModuleDestroyer>;
    using Pipeline = OptixHandle<OptixPipeline, PipelineDestroyer>;
    using ProgramGroup = OptixHandle<OptixProgramGroup, ProgramGroupDestroyer>;

    class GAS {
    public:
        GAS(
            optix::DeviceContext& context,
            int num_triangles,
            int num_vertices,
            float3 const *vertices,
            unsigned const *vertex_indices) {
            CHECK_CUDA(cudaMalloc((void**)&d_vertices, num_vertices * 3 * sizeof(float)));
            CHECK_CUDA(cudaMalloc((void**)&d_vertex_indices, num_triangles * 3 * sizeof(unsigned)));

            CHECK_CUDA(
                cudaMemcpy((void *)d_vertices, vertices, num_vertices * 3 * sizeof(float), cudaMemcpyHostToDevice)
            );
            CHECK_CUDA(
                cudaMemcpy((void *)d_vertex_indices, vertex_indices, num_triangles * 3 * sizeof(unsigned), cudaMemcpyHostToDevice)
            );

            OptixAccelBuildOptions accel_options = {};
            accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
            accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

            OptixBuildInput mesh_input = {};
            mesh_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
            mesh_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
            mesh_input.triangleArray.numVertices = num_vertices;
            mesh_input.triangleArray.vertexBuffers = &d_vertices;
            mesh_input.triangleArray.vertexStrideInBytes = sizeof(float3);

            mesh_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
            mesh_input.triangleArray.indexStrideInBytes = sizeof(unsigned) * 3;
            mesh_input.triangleArray.indexBuffer = d_vertex_indices;
            mesh_input.triangleArray.numIndexTriplets = num_triangles;

            mesh_input.triangleArray.preTransform = NULL;

            const uint32_t triangle_input_flags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
            mesh_input.triangleArray.flags = triangle_input_flags;

            mesh_input.triangleArray.numSbtRecords = 1;

            OptixAccelBufferSizes gas_buffer_sizes;
            CHECK_OPTIX(
                optixAccelComputeMemoryUsage(
                    context,
                    &accel_options,
                    &mesh_input,
                    1,
                    &gas_buffer_sizes);
            );

            CUdeviceptr d_temp_buffer_gas;
            CHECK_CUDA(
                cudaMalloc((void **)&d_gas_output_buffer, gas_buffer_sizes.outputSizeInBytes)
            );
            CHECK_CUDA(
                cudaMalloc((void **)&d_temp_buffer_gas, gas_buffer_sizes.tempSizeInBytes)
            );

            CHECK_OPTIX(
                optixAccelBuild(
                    context,
                    0,
                    &accel_options,
                    &mesh_input,
                    1,
                    d_temp_buffer_gas,
                    gas_buffer_sizes.tempSizeInBytes,
                    d_gas_output_buffer,
                    gas_buffer_sizes.outputSizeInBytes,
                    &traversable,
                    nullptr,
                    0
                )
            );

            CHECK_CUDA(cudaFree((void*)d_temp_buffer_gas));
        }

        GAS(GAS &&other) :
            traversable(0),
            d_vertices(0),
            d_vertex_indices(0),
            d_gas_output_buffer(0) {
            std::swap(traversable, other.traversable);
            std::swap(d_vertices, other.d_vertices);
            std::swap(d_vertex_indices, other.d_vertex_indices);
            std::swap(d_gas_output_buffer, other.d_gas_output_buffer);
        }

        GAS(GAS const &) = delete;
        void operator=(GAS const &) = delete;

        ~GAS() {
            // TODO: proper cleanup; is this enough? what do we do with the
            // traversable handle?
            CHECK_CUDA(cudaFree((void *)d_vertices));
            CHECK_CUDA(cudaFree((void *)d_vertex_indices));
            CHECK_CUDA(cudaFree((void *)d_gas_output_buffer));
        }

        OptixTraversableHandle handle() const {
            return traversable;
        }

    private:
        OptixTraversableHandle traversable;
        CUdeviceptr d_vertices;
        CUdeviceptr d_vertex_indices;
        CUdeviceptr d_gas_output_buffer;
    };

    class Instance_AS {
    public:
        Instance_AS(
            OptixTraversableHandle traversable,
            CUdeviceptr d_ias_output_buffer) :
            _traversable(traversable), 
            _d_ias_output_buffer(d_ias_output_buffer) {
        }

        Instance_AS(Instance_AS const &) = delete;
        void operator=(Instance_AS const &) = delete;

        ~Instance_AS() {
            // TODO: proper cleanup; is this enough? what do we do with the
            // traversable handle?
            CHECK_CUDA(cudaFree((void*)_d_ias_output_buffer));
        }

        Instance_AS &operator=(Instance_AS &&other) {
            cudaFree((void*)_d_ias_output_buffer);

            _traversable = 0;
            _d_ias_output_buffer = 0;
            std::swap(_traversable, other._traversable);
            std::swap(_d_ias_output_buffer, other._d_ias_output_buffer);

            return *this;
        }

        OptixTraversableHandle traversable_handle() {
            return _traversable;
        }

    private:
        OptixTraversableHandle _traversable;
        CUdeviceptr _d_ias_output_buffer;
    };

    template<typename ForwardIt>
    Instance_AS make_instance_AS(
        optix::DeviceContext &context,
        ForwardIt first, ForwardIt last) {

        OptixAccelBuildOptions accel_options = {};
        accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
        accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

        std::vector<OptixInstance> instances;

        std::transform(
            first, last,
            std::back_inserter(instances),
            [&](auto &mesh) -> OptixInstance {
                OptixInstance ret = {};
                memset(&ret, 0, sizeof(ret));

                float const *transform = mesh->transform();
                OptixTraversableHandle handle = mesh->traversable_handle();
                unsigned id = mesh->id();

                memcpy(ret.transform, transform, 12 * sizeof(float));
                ret.instanceId = id;
                ret.visibilityMask = 0xFF;
                ret.sbtOffset = 0;
                ret.flags = OPTIX_INSTANCE_FLAG_NONE;
                ret.traversableHandle = handle;

                return ret;
            }
        );
        
        auto count = instances.size();

        void *d_instances;
        CHECK_CUDA(cudaMalloc(&d_instances, count * sizeof(OptixInstance)));
        CHECK_CUDA(cudaMemcpy(d_instances, instances.data(), count * sizeof(OptixInstance), cudaMemcpyHostToDevice));

        OptixBuildInput build_input = {};
        build_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
        build_input.instanceArray.instances = (CUdeviceptr)d_instances;
        build_input.instanceArray.numInstances = count;

        OptixAccelBufferSizes ias_buffer_sizes;
        CHECK_OPTIX(
            optixAccelComputeMemoryUsage(
                context,
                &accel_options,
                &build_input,
                1,
                &ias_buffer_sizes);
        );

        CUdeviceptr d_ias_output_buffer;
        CUdeviceptr d_temp_buffer_ias;
        CHECK_CUDA(
            cudaMalloc((void **)&d_ias_output_buffer, ias_buffer_sizes.outputSizeInBytes)
        );
        CHECK_CUDA(
            cudaMalloc((void **)&d_temp_buffer_ias, ias_buffer_sizes.tempSizeInBytes)
        );

        OptixTraversableHandle traversable;
        CHECK_OPTIX(
            optixAccelBuild(
                context,
                0,
                &accel_options,
                &build_input,
                1,
                d_temp_buffer_ias,
                ias_buffer_sizes.tempSizeInBytes,
                d_ias_output_buffer,
                ias_buffer_sizes.outputSizeInBytes,
                &traversable,
                nullptr,
                0
            )
        );

        CHECK_CUDA(cudaFree((void*)d_instances));
        CHECK_CUDA(cudaFree((void*)d_temp_buffer_ias));

        return Instance_AS(traversable, d_ias_output_buffer);
    }

    class PipelineManager {
    public:
        PipelineManager(
            optix::Pipeline &&pipeline,
            optix::ProgramGroup &&pg_raygen,
            optix::ProgramGroup &&pg_hit,
            optix::ProgramGroup &&pg_miss,
            optix::Module &&module
        ) :
            _pipeline(std::move(pipeline)),
            _pg_raygen(std::move(pg_raygen)),
            _pg_hit(std::move(pg_hit)),
            _pg_miss(std::move(pg_miss)),
            _module(std::move(module)) {}

        optix::Pipeline _pipeline;
        optix::ProgramGroup _pg_raygen;
        optix::ProgramGroup _pg_hit;
        optix::ProgramGroup _pg_miss;
        optix::Module _module;
    };
}

