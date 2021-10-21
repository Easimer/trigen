// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: SDF AST to CUDA code generator
//

#include "../stdafx.h"
#include <cassert>
#include <cstdarg>
#include <array>
#include <softbody.h>
#include "../s_simulation.h"
#include "../types.h"
#define SB_BENCHMARK 1
#include "../s_benchmark.h"
#include "s_compute_cuda_codegen.h"
#include "cuda_utils.cuh"

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
                bufprintf("make_float3(%f, %f, %f)", buf[0], buf[1], buf[2]);
                break;
            case 4:
                bufprintf("make_float3(%f, %f, %f)", buf[0], buf[1], buf[2]);
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

    void do_visit(ast::Transform const& t) override {
        switch(t.kind()) {
            case ast::Transform::TRANSLATE:
                bufprintf("_translate(");
                break;
            //case ast::Primitive::ROTATION:
            //    bufprintf("_rotate(");
            //    break;
            default:
                assert(!"UNIMPLEMENTED TRANSFORM");
                break;
        }

        auto param_count = t.parameter_count();
        auto params = std::make_unique<ast::Node const*[]>(param_count);
        t.parameters(param_count, params.get());

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
        bufprintf("__device__ float scene(float3 const _sp) {\n");
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

static void dump_source_line_by_line(char* src) {
    int line = 1;

    printf("=== CUDA CODE DUMP ===\n");
    while(*src != 0) {
        auto cur = src;

        while(*cur != '\n') {
            cur++;
        }
        
        *cur = '\0';

        printf("%04d %s\n", line, src);

        *cur = '\n';
        line++;

        src = cur + 1;
    }

    printf("=== END OF CUDA CODE DUMP ===\n");
}

static sb::Unique_Ptr<char[]> generate_ptx(nvrtcProgram& prog, sb::sdf::ast::Expression<float>* expr) {
    std::vector<char> source_buffer;
    nvrtcResult rc;
    char const *name = "sb_sdf.cu.generated";

    include_sdf_library(source_buffer);
    generate_scene_function(source_buffer, expr);
    source_buffer.push_back('\0');

#ifndef NDEBUG
    char path[512];
    int t = time(NULL);
    snprintf(path, 511, "sb_sdf.generated_%d.cu", t);
    FILE* f = fopen(path, "wb");
    if(f != NULL) {
        fwrite(source_buffer.data(), source_buffer.size(), 1, f);
        fclose(f);
    }
#endif

    rc = nvrtcCreateProgram(&prog, source_buffer.data(), name, 0, NULL, NULL);
    if(rc != NVRTC_SUCCESS) {
        printf("sb: nvrtcCreateProgram failure, rc=%d, msg='%s'\n", rc, nvrtcGetErrorString(rc));
        return nullptr;
    }

#ifndef NDEBUG
    char const* nvcc_options[] = { "--device-debug", "--generate-line-info" };
    int nvcc_options_count = 2;
#else
    char const **nvcc_options = NULL;
    int nvcc_options_count = 0;
#endif

    rc = nvrtcCompileProgram(prog, nvcc_options_count, nvcc_options);
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

            dump_source_line_by_line(source_buffer.data());
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

    auto ret = std::make_unique<char[]>(ptx_len + 1);

    rc = nvrtcGetPTX(prog, ret.get());
    if(rc != NVRTC_SUCCESS) {
        printf("sb: nvrtcGetPTX failure, rc=%d, msg='%s'\n", rc, nvrtcGetErrorString(rc));
        nvrtcDestroyProgram(&prog);
        return nullptr;
    }

    ret[ptx_len] = '\0';

#ifdef DEBUG_DUMP_PTX
    FILE* f = fopen("/tmp/ptx_dump", "wb");
    fwrite(ret.get(), ptx_len, 1, f);
    fclose(f);
#endif
    return ret;
}

static bool upload_ptx(CUmodule& mod, int fun_count, CUfunction* funs[], char const* syms[], char const* ptx) {
    CUresult rc;

    rc = cuModuleLoadData(&mod, ptx);
    if(rc != CUDA_SUCCESS) {
        printf("sb: cuModuleLoadData failed: rc=%d\n", rc);
        return false;
    }

    for(int i = 0; i < fun_count; i++) {
        rc = cuModuleGetFunction(funs[i], mod, syms[i]);
        if(rc != CUDA_SUCCESS) {
            printf("sb: cuModuleGetFunction(%s) failed: rc=%d\n", syms[i], rc);
            return false;
        }
    }

    return true;
}

namespace sb::CUDA {
    struct AST_Program_Handle_ {
        CUmodule mod;
        nvrtcProgram prog;
        CUfunction k_gen_coll_constraints;
        CUfunction k_resolve_coll_constraints;
    };

    bool compile_ast(AST_Program_Handle* out_handle, sb::sdf::ast::Expression<float>* expr) {
        assert(out_handle != NULL);
        assert(expr != NULL);

        if(out_handle == NULL || expr == NULL) {
            return false;
        }

        nvrtcProgram prog;
        auto ptx = generate_ptx(prog, expr);
        if(!ptx) {
            return false;
        }

        CUmodule mod;
        CUfunction k_gen_coll_constraints, k_resolve_coll_constraints;
        CUfunction* funs[] = { &k_gen_coll_constraints, &k_resolve_coll_constraints };
        char const* syms[] = { "k_gen_coll_constraints", "k_resolve_coll_constraints" };
        CUresult rc;

        if(!upload_ptx(mod, 2, funs, syms, ptx.get())) {
            return false;
        }

        auto k = new AST_Program_Handle_;

        k->prog = prog;
        k->mod = mod;
        k->k_gen_coll_constraints = k_gen_coll_constraints;
        k->k_resolve_coll_constraints = k_resolve_coll_constraints;

        *out_handle = k;
        return true;
    }

    void free(AST_Program_Handle handle) {
        assert(handle != NULL);

        if(handle != NULL) {
            nvrtcDestroyProgram(&handle->prog);
            cuModuleUnload(handle->mod);

            delete handle;
        }
    }

    bool generate_collision_constraints(
            AST_Program_Handle handle,
            cudaStream_t stream,
            int N,
            CUDA_Array<unsigned char>& enable,
            CUDA_Array<float3>& intersections,
            CUDA_Array<float3>& normals,
            CUDA_Array<float4> const& predicted_positions,
            CUDA_Array<float4> const& positions,
            CUDA_Array<float> const& masses) {
        assert(handle != NULL);
        assert(!predicted_positions.is_empty());
        assert(!enable.is_empty());
        assert(!intersections.is_empty());
        assert(!normals.is_empty());
        assert(!positions.is_empty());
        assert(!masses.is_empty());

        if(handle == NULL || predicted_positions.is_empty() || positions.is_empty() || masses.is_empty() || enable.is_empty() || intersections.is_empty() || positions.is_empty()) {
            return false;
        }

        CUresult rc;
        auto const block_siz = 512;
        auto block_count = (N - 1) / block_siz + 1;
        int offset = 0;

        auto p_enable = (unsigned char*)enable;
        auto p_intersections = (float3*)intersections;
        auto p_normals = (float3*)normals;
        auto p_pred_pos = (float4 const*)predicted_positions;
        auto p_pos = (float4 const*)positions;
        auto p_masses = (float const*)masses;

        void* kparams[] = {
            &offset, &N, &p_enable, &p_intersections, &p_normals, &p_pred_pos, &p_pos, &p_masses
        };

        ASSERT_CUDA_SUCCEEDED(cuLaunchKernel(
            /* kernel: */ handle->k_gen_coll_constraints,
            /* grid:   */ block_count, 1, 1,
            /* block:  */ block_siz,   1, 1,
            /* smem:   */ 0,
            /* stream: */ stream,
            /* params: */ kparams,
            /* extra:  */ NULL
        ));

        return true;
    }

    bool resolve_collision_constraints(
            AST_Program_Handle handle,
            cudaStream_t stream,
            int N,
            CUDA_Array<float4>& predicted_positions,
            CUDA_Array<unsigned char> const& enable,
            CUDA_Array<float3> const& intersections,
            CUDA_Array<float3> const& normals,
            CUDA_Array<float4> const& positions,
            CUDA_Array<float> const& masses) {
        assert(handle != NULL);
        assert(!predicted_positions.is_empty());
        assert(!enable.is_empty());
        assert(!intersections.is_empty());
        assert(!normals.is_empty());
        assert(!positions.is_empty());
        assert(!masses.is_empty());

        assert(predicted_positions.N == N);
        assert(enable.N == N);
        assert(intersections.N == N);
        assert(normals.N == N);
        assert(positions.N == N);
        assert(masses.N == N);

        if(handle == NULL || predicted_positions.is_empty() || positions.is_empty() || masses.is_empty() || enable.is_empty() || intersections.is_empty() || positions.is_empty()) {
            return false;
        }

        CUresult rc;

        auto const block_siz = 512;
        auto block_count = (N - 1) / block_siz + 1;
        int offset = 0;

        auto p_enable = (unsigned char const*)enable;
        auto p_intersections = (float3 const*)intersections;
        auto p_normals = (float3 const*)normals;
        auto p_pred_pos = (float4*)predicted_positions;
        auto p_pos = (float4 const*)positions;
        auto p_masses = (float const*)masses;

        void* kparams[] = {
            &offset, &N, &p_pred_pos, &p_enable, &p_intersections, &p_normals, &p_pos, &p_masses
        };

        ASSERT_CUDA_SUCCEEDED(cuLaunchKernel(
            /* kernel: */ handle->k_resolve_coll_constraints,
            /* grid:   */ block_count, 1, 1,
            /* block:  */ block_siz,   1, 1,
            /* smem:   */ 0,
            /* stream: */ stream,
            /* params: */ kparams,
            /* extra:  */ NULL
        ));

        return true;
    }
}
