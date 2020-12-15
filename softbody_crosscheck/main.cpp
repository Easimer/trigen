// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: precompiled header
//

#include "stdafx.h"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <ctime>
#include "cross_check.h"
#include "benchmark.h"

class Stdio_Serializer : public sb::ISerializer {
public:
    Stdio_Serializer(FILE*&& f) : f(NULL) {
        std::swap(f, this->f);
        assert(this->f != NULL);
    }

    ~Stdio_Serializer() override {
        if(f != NULL) {
            fclose(f);
        }
    }

    size_t write(void const* ptr, size_t size) override {
        return fwrite(ptr, 1, size, f);
    }

    void seek_to(size_t fp) override {
        fseek(f, fp, SEEK_SET);
    }

    void seek(int off) override {
        fseek(f, off, SEEK_CUR);
    }

    size_t tell() override {
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
        printf("\tReference orientation: [%f %f %f %f]\n", ref.orientation.x, ref.orientation.y, ref.orientation.z, ref.orientation.w);
        printf("\tOther orientation:     [%f %f %f %f]\n", other.orientation.x, other.orientation.y, other.orientation.z, other.orientation.w);
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

static int find_arg_flag(int argc, char** argv, char const* flag, bool* out) {
    *out = false;
    for(int i = 1; i < argc; i++) {
        if(strcmp(argv[i], flag) == 0) {
            *out = true;
            return i;
        }
    }

    return -1;
}

static int run_crosscheck(int argc, char** argv) {
    printf("mode: cross-check\n");

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

static int run_benchmark(int argc, char** argv) {
    printf("mode: benchmark\n");
    bool _;
    int dim = 64;

    auto size_flag_idx = find_arg_flag(argc, argv, "-s", &_);
    if(size_flag_idx != -1 && argc > size_flag_idx + 1) {
        sscanf(argv[size_flag_idx + 1], "%d", &dim);
    }

    auto backend_flag_idx = find_arg_flag(argc, argv, "-B", &_);
    assert(backend_flag_idx != -1);
    if(argc > backend_flag_idx + 1) {
        int backend_idx = 0;
        sscanf(argv[backend_flag_idx + 1], "%d", &backend_idx);
        printf("mode: benchmark backend=%d dim=%d\n", backend_idx, dim);
        auto B = Benchmark::make_benchmark((sb::Compute_Preference)backend_idx, dim);

        B.run(2.5f);
    } else {
        printf("argument -B requires a parameter (compute backend id like 1, 2 or 3)\n");
    }

    return 0;
}

int main(int argc, char** argv) {
    bool should_run_benchmark = false;

    find_arg_flag(argc, argv, "-B", &should_run_benchmark);

    if(should_run_benchmark) {
        return run_benchmark(argc, argv);
    } else {
        return run_crosscheck(argc, argv);
    }
}
