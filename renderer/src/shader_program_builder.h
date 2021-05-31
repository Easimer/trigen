// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: OpenGL shader program builder
//

#pragma once

#include <string>

#include "glres.h"

namespace gl {

class Shader_Program_Builder {
public:
    Shader_Program_Builder();
    Shader_Program_Builder(std::string const &name);
    Shader_Program_Builder &Attach(Vertex_Shader const &vsh);
    Shader_Program_Builder &Attach(Fragment_Shader const &fsh);

    Optional<Shader_Program> Link();

    char const *Error() const;
    char const *Name() const;

private:
    Shader_Program _program;
    std::string _name;
    std::string _errorMsg;
};

}
