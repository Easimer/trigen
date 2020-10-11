// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: SDF AST to CUDA code generator
//

namespace sb::CUDA {
    typedef struct AST_Kernel_Handle_ *AST_Kernel_Handle;

    bool compile_ast(AST_Kernel_Handle* out_handle, sb::sdf::ast::Expression<float>* expr);
    void free(AST_Kernel_Handle handle);
    bool exec(AST_Kernel_Handle handle, int N, float* distances, Vec4 const* sample_points);

    struct AST_Kernel {
        AST_Kernel_Handle handle;

        AST_Kernel() : handle(nullptr) {}
        AST_Kernel(AST_Kernel_Handle handle) : handle(handle) {}
        AST_Kernel(AST_Kernel const&) = delete;
        AST_Kernel(AST_Kernel&& other) : handle(nullptr) {
            std::swap(handle, other.handle);
        }

        AST_Kernel& operator=(AST_Kernel&& other) {
            if(handle != nullptr) {
                free(handle);
                handle = nullptr;
            }

            std::swap(handle, other.handle);

            return *this;
        }
    };
}
