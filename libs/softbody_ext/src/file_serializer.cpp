// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: FILE* based serializer for `softbody`
//

#include <softbody.h>
#include <memory>
#include <cstdio>
#include <softbody/file_serializer.h>

class Impl : public sb::ISerializer, public sb::IDeserializer {
public:
    Impl(FILE* f) : f(f) {}
    ~Impl() {
        fclose(f);
    }

private:
    FILE* f;

    size_t write(void const* ptr, size_t size) override {
        auto res = fwrite(ptr, 1, size, f);
        if (res < 0) res = 0;
        return res;
    }

    void seek_to(size_t file_point) override {
        fseek(f, file_point, SEEK_SET);
    }
    
    void seek(int offset) override {
        fseek(f, offset, SEEK_CUR);
    }
    
    size_t tell() override {
        return ftell(f);
    }

    size_t read(void* ptr, size_t size) override {
        auto res = fread(ptr, 1, size, f);
        if (res < 0) res = 0;
        return res;
    }
};

namespace sb {
    std::unique_ptr<sb::ISerializer> make_serializer(char const *path) {
        auto f = fopen(path, "wb");

        if (f != NULL) {
            return std::make_unique<Impl>(f);
        } else {
            return nullptr;
        }
    }

    std::unique_ptr<sb::IDeserializer> make_deserializer(char const *path) {
        auto f = fopen(path, "rb");

        if (f != NULL) {
            return std::make_unique<Impl>(f);
        } else {
            return nullptr;
        }
    }
}
