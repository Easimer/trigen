// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: virtual filesystem
//

#include <fstream>
#include <sstream>
#include <optional>
#include <unordered_map>

#include "virtfs.hpp"

struct Entry {
    char const *resource;
    std::optional<std::string> onDisk;
};

using Entry_Map = std::unordered_map<std::string, Entry>;
static Entry_Map *files = nullptr;

namespace virtfs {

static void Initialize() {
    if(files != nullptr) return;
    files = new Entry_Map;
}

Status GetFile(std::string const &path, char const **contents) {
    if(files == nullptr) {
        Initialize();
    }

    if (files->count(path) == 0) {
        return Status::NotFound;
    }

    if (contents == nullptr) {
        return Status::InvalidArguments;
    }

    auto &entry = files->at(path);

    if (entry.onDisk) {
        *contents = entry.onDisk->c_str();
    } else {
        *contents = entry.resource;
    }

    return Status::Ok;
}

Status RegisterFile(std::string const &path, char const *contents) {
    if(files == nullptr) {
        Initialize();
    }

    if (contents == nullptr || path.length() == 0) {
        return Status::InvalidArguments;
    }

    if (files->count(path) != 0) {
        return Status::Duplicate;
    }

    Entry entry;
    entry.resource = contents;

    std::ifstream ifs(path);
    if (ifs) {
        std::stringstream ss;
        ss << ifs.rdbuf();
        entry.onDisk = std::move(ss.str());

        printf("[ virtfs ] file '%s' found on disk, replacing the in-memory version\n", path.c_str());
    }

    (*files)[path] = std::move(entry);

    return Status::Ok;
}

}
