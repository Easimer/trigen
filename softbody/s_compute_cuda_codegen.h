// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: SDF AST to CUDA code generator
//

#include "cuda_utils.cuh"

namespace sb::CUDA {
    typedef struct AST_Program_Handle_ *AST_Program_Handle;

    bool compile_ast(AST_Program_Handle* out_handle, sb::sdf::ast::Expression<float>* expr);
    void free(AST_Program_Handle handle);

    bool generate_collision_constraints(
            AST_Program_Handle handle,
            cudaStream_t stream,
            int N,
            CUDA_Array<unsigned char>& enable,
            CUDA_Array<float3>& intersections,
            CUDA_Array<float3>& normals,
            CUDA_Array<float4> const& predicted_positions,
            CUDA_Array<float4> const& positions,
            CUDA_Array<float> const& masses);

    bool resolve_collision_constraints(
            AST_Program_Handle handle,
            cudaStream_t stream,
            int N,
            CUDA_Array<float4>& predicted_positions,
            CUDA_Array<unsigned char> const& enable,
            CUDA_Array<float3> const& intersections,
            CUDA_Array<float3> const& normals,
            CUDA_Array<float4> const& positions,
            CUDA_Array<float> const& masses);

    struct AST_Program {
        AST_Program_Handle handle;

        AST_Program() : handle(nullptr) {}
        AST_Program(AST_Program_Handle handle) : handle(handle) {}
        AST_Program(AST_Program const&) = delete;
        AST_Program(AST_Program&& other) : handle(nullptr) {
            std::swap(handle, other.handle);
        }

        AST_Program& operator=(AST_Program&& other) {
            if(handle != nullptr) {
                free(handle);
                handle = nullptr;
            }

            std::swap(handle, other.handle);

            return *this;
        }

        ~AST_Program() {
            if(handle != nullptr) {
                free(handle);
            }
        }

        operator AST_Program_Handle() const {
            return handle;
        }
    };

    using AST_Kernel = AST_Program;

    using AST_Kernel_Handle = AST_Program_Handle;
}
