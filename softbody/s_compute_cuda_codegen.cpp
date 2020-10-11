// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: SDF AST to CUDA code generator
//

#include "common.h"
#include <cassert>
#include <cstdarg>
#include <array>
#include <softbody.h>
#include "s_simulation.h"
#define SB_BENCHMARK 1
#include "s_benchmark.h"
#include "s_compute_cuda_codegen.h"

#include <cuda.h>
#include <nvrtc.h>

extern "C" char const* cuda_templates_cu;
extern "C" unsigned long long const cuda_templates_cu_len;

using namespace sb::sdf;

class CUDA_Codegen_Visitor : public ast::Visitor {
public:
    CUDA_Codegen_Visitor(std::vector<char>& buffer) : _buffer(buffer) {}

    void do_visit(ast::Sample_Point const& sp) override {
        bufprintf("_sp");
    }

    void do_visit(ast::Base_Vector_Constant const& v, size_t len) override {
        auto buf = std::make_unique<float[]>(len);
        v.value(buf.get());

        switch(len) {
            case 1:
                bufprintf("%f", buf[0]);
                break;
            case 2:
                bufprintf("make_float2(%f, %f)", buf[0], buf[1]);
                break;
            case 3:
                bufprintf("make_float4(%f, %f, %f, 0)", buf[0], buf[1], buf[2]);
                break;
            case 4:
                bufprintf("make_float4(%f, %f, %f, %f)", buf[0], buf[1], buf[2], buf[3]);
                break;
        }
    }

    void do_visit(ast::Primitive const& p) override {
        switch(p.kind()) {
            case ast::Primitive::UNION:
                bufprintf("_union(");
                break;
            case ast::Primitive::SUBTRACTION:
                bufprintf("_subtract(");
                break;
            case ast::Primitive::INTERSECTION:
                bufprintf("_intersect(");
                break;
            case ast::Primitive::BOX:
                bufprintf("_box(");
                break;
            case ast::Primitive::SPHERE:
                bufprintf("_sphere(");
                break;
            default:
                assert(!"UNIMPLEMENTED PRIMITIVE");
                break;
        }

        auto param_count = p.parameter_count();
        auto params = std::make_unique<ast::Node const*[]>(param_count);
        p.parameters(param_count, params.get());

        for(size_t i = 0; i < param_count; i++) {
            params[i]->visit(this);

            if(i != param_count - 1) {
                bufprintf(", ");
            }
        }
        bufprintf(")");
    }

    void bufprintf(char const* format, ...) {
        va_list ap;
        int size;

        va_start(ap, format);
        size = vsnprintf(NULL, 0, format, ap);
        va_end(ap);

        if(size < 0) {
            return;
        }

        size++;
        auto buf = std::make_unique<char[]>(size);

        va_start(ap, format);
        size = vsnprintf(buf.get(), 127, format, ap);
        va_end(ap);

        _buffer.insert(_buffer.end(), buf.get(), buf.get() + size);
    }

    void begin_sdf_function() {
        bufprintf("__device__ float scene(float4 const _sp) {\n");
        bufprintf("    return ");
    }

    void end_sdf_function() {
        bufprintf(";\n}\n");
    }

private:
    std::vector<char>& _buffer;
};

static void include_sdf_library(std::vector<char>& ret) {
    auto start = (char const*)cuda_templates_cu;
    auto end = start + cuda_templates_cu_len - 1;
    ret.insert(ret.end(), start, end);
}

static void generate_scene_function(std::vector<char>& ret, sb::sdf::ast::Expression<float>* expr) {
    CUDA_Codegen_Visitor visitor(ret);
    visitor.begin_sdf_function();
    expr->visit(&visitor);
    visitor.end_sdf_function();
}

static sb::Unique_Ptr<char[]> generate_ptx(sb::sdf::ast::Expression<float>* expr) {
    std::vector<char> source_buffer;
    nvrtcResult rc;
    nvrtcProgram prog;
    char const *name = "sb_sdf.cu.generated";

    include_sdf_library(source_buffer);
    generate_scene_function(source_buffer, expr);
    source_buffer.push_back('\0');

    rc = nvrtcCreateProgram(&prog, source_buffer.data(), name, 0, NULL, NULL);
    if(rc != NVRTC_SUCCESS) {
        printf("sb: nvrtcCreateProgram failure, rc=%d, msg='%s'\n", rc, nvrtcGetErrorString(rc));
        return nullptr;
    }

    rc = nvrtcCompileProgram(prog, 0, NULL);
    if(rc != NVRTC_SUCCESS) {
        printf("sb: nvrtcCompileProgram failure, rc=%d, msg='%s'\n", rc, nvrtcGetErrorString(rc));

        if(rc == NVRTC_ERROR_COMPILATION) {
            size_t log_size;
            nvrtcGetProgramLogSize(prog, &log_size);
            std::vector<char> log_buf;
            log_buf.resize(log_size + 1);
            nvrtcGetProgramLog(prog, log_buf.data());
            log_buf[log_size] = '\0';
            printf("\tLog:\n%s\n=== END OF COMPILE LOG ===\n", log_buf.data());
        }

        nvrtcDestroyProgram(&prog);
        return nullptr;
    }

    size_t ptx_len;
    rc = nvrtcGetPTXSize(prog, &ptx_len);
    if(rc != NVRTC_SUCCESS) {
        printf("sb: nvrtcGetPTXSize failure, rc=%d, msg='%s'\n", rc, nvrtcGetErrorString(rc));
        nvrtcDestroyProgram(&prog);
        return nullptr;
    }

    auto ret = std::make_unique<char[]>(ptx_len);

    rc = nvrtcGetPTX(prog, ret.get());
    if(rc != NVRTC_SUCCESS) {
        printf("sb: nvrtcGetPTX failure, rc=%d, msg='%s'\n", rc, nvrtcGetErrorString(rc));
        nvrtcDestroyProgram(&prog);
        return nullptr;
    }

    nvrtcDestroyProgram(&prog);
    return ret;
}

static bool upload_ptx(CUmodule& mod, CUfunction& fun, char const* ptx) {
    CUresult rc;

    rc = cuModuleLoadData(&mod, ptx);
    if(rc != CUDA_SUCCESS) {
        printf("sb: cuModuleLoadData failed: rc=%d\n", rc);
        return false;
    }

    rc = cuModuleGetFunction(&fun, mod, "k_exec_sdf");
    if(rc != CUDA_SUCCESS) {
        printf("sb: cuModuleGetFunction failed: rc=%d\n", rc);
        return false;
    }

    return true;
}

namespace sb::CUDA {
    struct AST_Kernel_Handle_ {
        CUmodule mod;
        CUfunction fun;
    };

    bool compile_ast(AST_Kernel_Handle* out_handle, sb::sdf::ast::Expression<float>* expr) {
        assert(out_handle != NULL);
        assert(expr != NULL);

        if(out_handle == NULL || expr == NULL) {
            return false;
        }

        auto ptx = generate_ptx(expr);
        if(!ptx) {
            return false;
        }

        CUmodule mod;
        CUfunction fun;
        CUresult rc;

        if(!upload_ptx(mod, fun, ptx.get())) {
            return false;
        }

        auto k = new AST_Kernel_Handle_;

        k->mod = mod;
        k->fun = fun;

        *out_handle = k;
        return true;
    }

    void free(AST_Kernel_Handle handle) {
        assert(handle != NULL);

        if(handle != NULL) {
            cuModuleUnload(handle->mod);

            delete handle;
        }
    }

    bool exec(AST_Kernel_Handle handle, int N, float* distances, Vec4 const* sample_points) {
        assert(handle != NULL);
        assert(sample_points != NULL);
        assert(distances != NULL);

        if(handle == NULL || sample_points == NULL || distances == NULL) {
            return false;
        }

        CUresult rc;
        CUdeviceptr d_sample_points, d_distances;

        rc = cuMemAlloc(&d_sample_points, N * 4 * sizeof(float));
        if(rc != CUDA_SUCCESS) {
            printf("sb: cuMemAlloc(d_sample_points) failed, rc=%d\n", rc);
            return false;
        }

        rc = cuMemAlloc(&d_distances, N * sizeof(float));
        if(rc != CUDA_SUCCESS) {
            printf("sb: cuMemAlloc(d_distances) failed, rc=%d\n", rc);
            cuMemFree(d_sample_points);
            return false;
        }

        rc = cuMemcpyHtoD(d_sample_points, sample_points, N * 4 * sizeof(float));
        if(rc != CUDA_SUCCESS) {
            printf("sb: cuMemcpyHtoD(d_sample_points, sample_points) failed, rc=%d\n", rc);
            cuMemFree(d_distances);
            cuMemFree(d_sample_points);
            return false;
        }

        auto const block_siz = 512;
        auto block_count = (N - 1) / block_siz + 1;
        int offset = 0;

        void* kparams[] = {
            &offset, &N, &d_distances, &d_sample_points
        };

        rc = cuLaunchKernel(
            /* kernel: */ handle->fun,
            /* grid:   */ block_count, 1, 1,
            /* block:  */ block_siz,   1, 1,
            /* smem:   */ 0,
            /* stream: */ 0,
            /* params: */ kparams,
            /* extra:  */ NULL
        );
        if(rc != CUDA_SUCCESS) {
            printf("sb: cuLaunchKernel(k_exec_sdf) failed, rc=%d\n", rc);
            cuMemFree(d_distances);
            cuMemFree(d_sample_points);
            return false;
        }

        rc = cuMemFree(d_sample_points);
        if(rc != CUDA_SUCCESS) {
            printf("sb: cuMemFree(d_sample_points) failed, rc=%d\n", rc);
            cuMemFree(d_distances);
            cuMemFree(d_sample_points);
            return false;
        }

        rc = cuMemcpyDtoH(distances, d_distances, N * sizeof(float));
        if(rc != CUDA_SUCCESS) {
            printf("sb: cuMemcpyDtoH(distances, d_distances) failed, rc=%d\n", rc);
            cuMemFree(d_distances);
            return false;
        }

        cuMemFree(d_distances);
        return true;
    }
}
