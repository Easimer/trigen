// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: virtual filesystem
//

#pragma once

#include <string>

namespace virtfs {

enum class Status {
    Ok,
    NotFound,
    InvalidArguments,
    Duplicate,
};

Status GetFile(std::string const &path, char const **contents);
Status RegisterFile(std::string const &path, char const *contents);

struct Startup_Registrar {
    Startup_Registrar(char const *path, char const *contents) {
        RegisterFile(path, contents);
    }
};

#define _VIRTFS_GENERATE_VARIABLE2(l) _virtfs_reg_ ## l
#define _VIRTFS_GENERATE_VARIABLE(l) _VIRTFS_GENERATE_VARIABLE2(l)
#define VIRTFS_REGISTER_RESOURCE(path, contents) static virtfs::Startup_Registrar _VIRTFS_GENERATE_VARIABLE(__LINE__)(path, contents)
}
