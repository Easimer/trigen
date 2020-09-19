// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: precompiled header
//

#include "stdafx.h"
#include <cassert>
#include <cstdio>
#include "cross_check.h"

class Stdio_Serializer : public sb::ISerializer {
public:
    Stdio_Serializer(FILE*&& f) : f(NULL) {
        std::swap(f, this->f);
        assert(this->f != NULL);
    }

    virtual ~Stdio_Serializer() {
        if(f != NULL) {
            fclose(f);
        }
    }

    size_t write(void const* ptr, size_t size) override {
        return fwrite(ptr, 1, size, f);
    }

    void seek_to(size_t fp) {
        fseek(f, fp, SEEK_SET);
    }

    void seek(int off) {
        fseek(f, off, SEEK_CUR);
    }

    size_t tell() {
        return ftell(f);
    }

private:
    FILE* f;
};

class CCL : public Cross_Check_Listener {
public:
    CCL() :
        error(false) {
    }

    bool is_error() {
        return error;
    }
private:
    bool error;
    void on_fault(
            sb::Compute_Preference backend,
            sb::index_t pidx,
            sb::Particle ref,
            sb::Particle other,
            char const* message,
            char const* step) override {
        error = true;
        printf("======================================\n");
        printf("ERROR DETECTED\n");
        printf("Backend: %d\n", backend);
        printf("In step: '%s'\n", step);
        printf("Particle index: %zd\n", pidx);
        printf("\tReference particle: [%f %f %f]\n", ref.position.x, ref.position.y, ref.position.z);
        printf("\tOther particle:     [%f %f %f]\n", other.position.x, other.position.y, other.position.z);
        printf("Error message:\n\t\"%s\"\n", message);
    }
};

static void dump_images(Cross_Check& cc) {
    char path_template_buf[128];
    char path_buf1[128];
    char path_buf2[128];
    char path_buf3[128];
    snprintf(path_template_buf, 127, "dump_%zd_%%d.simg", time(NULL));
    path_template_buf[127] = '\0';

    snprintf(path_buf1, 127, path_template_buf, 1);
    snprintf(path_buf2, 127, path_template_buf, 2);
    snprintf(path_buf3, 127, path_template_buf, 3);

    Stdio_Serializer fio[3] = {
        Stdio_Serializer(fopen(path_buf1, "wb")),
        Stdio_Serializer(fopen(path_buf2, "wb")),
        Stdio_Serializer(fopen(path_buf3, "wb")),
    };

    sb::ISerializer* io[3] = { &fio[0], &fio[1], &fio[2] };
    cc.dump_images(io);
}

int main(int argc, char** argv) {
    CCL ccl;
    Cross_Check cc;

    while(!ccl.is_error()) {
        cc.step(&ccl);
    }


    if(ccl.is_error()) {
        dump_images(cc);
    }

    return 0;
}
