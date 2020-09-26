// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: SDF AST to CUDA code generator
//

#include <cassert>
#include <cstdarg>
#include <array>
#include <softbody.h>
#define SB_BENCHMARK 1
#include "s_benchmark.h"

extern char const* cuda_templates_cu;
extern unsigned long long cuda_templates_cu_len;

using namespace sb::sdf;

class CUDA_Codegen_Visitor : public ast::Visitor {
public:
    CUDA_Codegen_Visitor(std::vector<char>& buffer) : _buffer(buffer) {}

    void do_visit(ast::Sample_Point const& sp) override {
        char const sp_sym[] = "_sp";
        _buffer.insert(_buffer.end(), std::begin(sp_sym), std::end(sp_sym));
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
                bufprintf("union(");
                break;
            case ast::Primitive::SUBTRACTION:
                bufprintf("subtract(");
                break;
            case ast::Primitive::INTERSECTION:
                bufprintf("intersect(");
                break;
            case ast::Primitive::BOX:
                bufprintf("box(");
                break;
            case ast::Primitive::SPHERE:
                bufprintf("sphere(");
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
        size = vsnprintf(NULL, 127, format, ap);
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

private:
    std::vector<char>& _buffer;
};

std::vector<char> translate_sdf_ast_to_cuda(sb::sdf::ast::Expression<float>* expr) {
    std::vector<char> ret;
    CUDA_Codegen_Visitor visitor(ret);
    return ret;
}
