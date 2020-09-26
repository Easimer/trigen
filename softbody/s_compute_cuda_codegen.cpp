// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: SDF AST to CUDA code generator
//

#include "common.h"
#include <cassert>
#include <cstdarg>
#include <array>
#include <softbody.h>
#define SB_BENCHMARK 1
#include "s_benchmark.h"

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
        bufprintf("__device__ float scene(float4 const _sp) {\n    return ");
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

std::vector<char> generate_kernel(sb::sdf::ast::Expression<float>* expr) {
    std::vector<char> ret;

    include_sdf_library(ret);
    generate_scene_function(ret, expr);

    return ret;
}
