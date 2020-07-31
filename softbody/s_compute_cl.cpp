// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: OpenCL computation backend
//

#include "stdafx.h"
#include <cassert>
#include <array>
#include <algorithm>
#include <optional>
#include "softbody.h"
#include "l_iterators.h"
#include "s_compute_backend.h"
#define SB_BENCHMARK (1)
#include "s_benchmark.h"
#include "m_utils.h"
#include <glm\gtx\matrix_operation.hpp>

#define __CL_ENABLE_EXCEPTIONS
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif
#include <glm\gtc\type_ptr.hpp>

#define NUMBER_OF_CLUSTERS(idx) (s.edges[(idx)].size() + 1)

class calc_mass;

class Compute_CL : public ICompute_Backend {
public:
    Compute_CL(cl::Context&& _ctx, cl::Program&& _program)
        : ctx(std::move(_ctx)), program(std::move(_program)), queue(ctx)
    {
        assert(sanity_check_mat_mul());
        assert(sanity_check_mueller_rotation_extraction());
    }
private:
    cl::Context ctx;
    cl::Program program;
    cl::CommandQueue queue;

    cl::Buffer d_A, d_invRest, d_q;

    float mass_of_particle(System_State const& s, unsigned i) const {
        auto const d_i = s.density[i];
        auto const s_i = s.size[i];
        auto const m_i = (4.f / 3.f) * glm::pi<float>() * s_i.x * s_i.y * s_i.z * d_i;
        return m_i;
    }

    size_t particle_count(System_State const& s) const {
        return s.position.size();
    }

    void begin_new_frame(System_State const& sim) override {
        auto const N = particle_count(sim);
        d_A = cl::Buffer(ctx, CL_MEM_READ_ONLY, N * 16 * sizeof(float));
        d_invRest = cl::Buffer(ctx, CL_MEM_READ_ONLY, N * 16 * sizeof(float));
        d_q = cl::Buffer(ctx, CL_MEM_READ_WRITE, N * 4 * sizeof(float));
    }

    Vector<float> calculate_particle_masses(System_State & s) {
        auto N = particle_count(s);
        Vector<float> h_masses;
        Vector<glm::vec4> h_sizes;

        h_masses.resize(N);
        h_sizes.resize(N);

        for (auto i = 0ull; i < N; i++) {
            h_sizes[i] = glm::vec4(s.size[i], 0);
        }

        cl::Buffer d_sizes(ctx, h_sizes.begin(), h_sizes.end(), true);
        cl::Buffer d_densities(ctx, s.density.begin(), s.density.end(), true);
        cl::Buffer d_masses(ctx, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, N * sizeof(float));

        cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer> kernel(program, "calculate_particle_masses");

        queue.enqueueWriteBuffer(d_densities, CL_FALSE, 0, N * sizeof(float), s.density.data());
        queue.enqueueWriteBuffer(d_sizes, CL_FALSE, 0, N * 4 * sizeof(float), h_sizes.data());

        kernel(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(N), cl::NDRange(1)), d_sizes, d_densities, d_masses);

        queue.enqueueReadBuffer(d_masses, true, 0, N * sizeof(float), h_masses.data());

        return h_masses;
    }

    bool sanity_check_mueller_rotation_extraction() {
        auto qx = glm::angleAxis(glm::degrees(90.0f), glm::vec3(1, 0, 0));
        auto qy = glm::angleAxis(glm::degrees(90.0f), glm::vec3(0, 1, 0));
        auto qz = glm::angleAxis(glm::degrees(90.0f), glm::vec3(0, 0, 1));

        Quat quaternions[3] = { qx, qy, qz };
        glm::mat4 matrices[3] = { glm::mat3(qx), glm::mat3(qy), glm::mat3(qz) };
        Quat approx[3] = { glm::identity<Quat>(), glm::identity<Quat>(), glm::identity<Quat>() };
        Quat expected[3] = { glm::identity<Quat>(), glm::identity<Quat>(), glm::identity<Quat>() };

        for (int i = 0; i < 3; i++) {
            mueller_rotation_extraction(Mat3(matrices[i]), expected[i]);
        }

        cl::Buffer d_A(ctx, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY, 3 * 16 * sizeof(float));
        cl::Buffer d_q(ctx, CL_MEM_READ_WRITE, 3 * 4 * sizeof(float));

        cl::make_kernel<cl::Buffer, cl::Buffer> kernel(program, "mueller_rotation_extraction");

        queue.enqueueWriteBuffer(d_A, CL_FALSE, 0, 3 * 16 * sizeof(float), matrices);
        queue.enqueueWriteBuffer(d_q, CL_FALSE, 0, 3 *  4 * sizeof(float), approx);

        kernel(cl::EnqueueArgs(queue, cl::NDRange(3)), d_A, d_q);

        queue.enqueueReadBuffer(d_q, CL_TRUE, 0, 3 * 4 * sizeof(float), approx);

        bool ret = true;
        for (int i = 0; i < 3; i++) {
            auto e = glm::normalize(expected[i]);
            auto a = glm::normalize(approx[i]);

            if (glm::length(e - a) > 4 * glm::epsilon<float>()) {
                printf("sb: MUELLER_ROTATION_EXTRACTION SANITY CHECK FAILED!\n");
                printf("\texpected: %f %f %f %f  got: %f %f %f %f\n",
                    e[0], e[1], e[2], e[3],
                    a[0], a[1], a[2], a[3]);
                ret = false;
            }
        }
        return ret;
    }

    bool sanity_check_mat_mul() {
        glm::mat4 lhs;
        glm::mat4 rhs;
        glm::mat4 out;

        for (int i = 0; i < 4; i++) {
            lhs[i][0] = rand() % 128 - 64;
            lhs[i][1] = rand() % 128 - 64;
            lhs[i][2] = rand() % 128 - 64;
            lhs[i][3] = rand() % 128 - 64;
            rhs[i][0] = rand() % 128 - 64;
            rhs[i][1] = rand() % 128 - 64;
            rhs[i][2] = rand() % 128 - 64;
            rhs[i][3] = rand() % 128 - 64;
        }

        auto const expected = lhs * rhs;

        cl::Buffer d_lhs(ctx, glm::value_ptr(lhs), glm::value_ptr(lhs) + 16, true);
        cl::Buffer d_rhs(ctx, glm::value_ptr(rhs), glm::value_ptr(rhs) + 16, true);
        cl::Buffer d_out(ctx, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, 16 * sizeof(float));

        cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer> kernel(program, "mat_mul_main");

        queue.enqueueWriteBuffer(d_lhs, CL_FALSE, 0, 16 * sizeof(float), glm::value_ptr(lhs));
        queue.enqueueWriteBuffer(d_rhs, CL_FALSE, 0, 16 * sizeof(float), glm::value_ptr(rhs));

        kernel(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(1), cl::NDRange(1)), d_out, d_lhs, d_rhs);

        queue.enqueueReadBuffer(d_out, CL_TRUE, 0, 16 * sizeof(float), glm::value_ptr(out));

        auto diff_mat = expected - out;
        auto diff = glm::determinant(diff_mat);

        if (glm::abs(diff) > 4 * glm::epsilon<float>()) {
            printf("sb: MAT_MUL SANITY CHECK FAILED!\n");

            printf("Output:\n");
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    printf("%f ", out[j][i]);
                }
                printf("\n");
            }
            printf("\n");

            printf("Expected:\n");
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    printf("%f ", expected[j][i]);
                }
                printf("\n");
            }
            printf("\n");

            return false;
        }

        return true;
    }

    void do_one_iteration_of_shape_matching_constraint_resolution(
        System_State& s,
        float phdt
    ) override {
        DECLARE_BENCHMARK_BLOCK();
        BEGIN_BENCHMARK();

        auto const N = particle_count(s);
        std::vector<glm::mat4> h_A;
        std::vector<glm::mat4> h_invRest;
        h_A.reserve(N);
        h_invRest.reserve(N);

        for (unsigned i = 0; i < N; i++) {
            std::array<unsigned, 1> me{ i };
            auto& neighbors = s.edges[i];
            auto neighbors_and_me = iterator_union(neighbors.begin(), neighbors.end(), me.begin(), me.end());

            // Sum particle weights in the current cluster
            auto M = std::accumulate(
                neighbors.begin(), neighbors.end(),
                mass_of_particle(s, i),
                [&](float acc, unsigned idx) {
                    return acc + mass_of_particle(s, idx);
                }
            );

            assert(M != 0);

            auto invRest = s.bind_pose_inverse_bind_pose[i];
            auto com0 = s.bind_pose_center_of_mass[i];

            // Center of mass calculated using the predicted positions
            auto com_cur = std::accumulate(
                neighbors.begin(), neighbors.end(),
                mass_of_particle(s, i) * s.predicted_position[i],
                [&](Vec3 const& acc, unsigned idx) {
                    return acc + mass_of_particle(s, idx) * s.predicted_position[idx];
                }
            ) / M;

            s.center_of_mass[i] = com_cur;

            // Calculates the moment matrix of a single particle
            auto calc_A_i = [&](unsigned i) -> Mat3 {
                auto m_i = mass_of_particle(s, i);
                auto A_i = 1.0f / 5.0f * glm::diagonal3x3(s.size[i] * s.size[i]) * Mat3(s.orientation[i]);

                return m_i * (A_i + glm::outerProduct(s.predicted_position[i], s.bind_pose[i]) - glm::outerProduct(com_cur, com0));
            };

            // Calculate the cluster moment matrix
            auto A = std::accumulate(
                neighbors.begin(), neighbors.end(),
                calc_A_i(i),
                [&](Mat3 const& acc, unsigned idx) -> Mat3 {
                    return acc + calc_A_i(idx);
                }
            );

            h_invRest.push_back(glm::mat4(invRest));
            h_A.push_back(glm::mat4(A));
        }

        cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer> kernel(program, "calculate_optimal_rotation");

        queue.enqueueWriteBuffer(d_A, CL_FALSE, 0, N * 16 * sizeof(float), h_A.data());
        queue.enqueueWriteBuffer(d_invRest, CL_FALSE, 0, N * 16 * sizeof(float), h_invRest.data());
        queue.enqueueWriteBuffer(d_q, CL_FALSE, 0, N *  4 * sizeof(float), s.predicted_orientation.data());

        kernel(cl::EnqueueArgs(queue, cl::NDRange(N)), d_A, d_invRest, d_q);

        struct Particle_Correction_Info {
            Vec3 pos_bind;
            Vec3 com_cur;
            float inv_numClusters;
        };

        Vector<Particle_Correction_Info> correction_infos;
        correction_infos.reserve(N);

        // Calculate what we can while we wait for the extracted quaternions
        for (unsigned i = 0; i < N; i++) {
            auto const com0 = s.bind_pose_center_of_mass[i];
            correction_infos.push_back({});
            auto& inf = correction_infos.back();
            inf.pos_bind = s.bind_pose[i] - com0;
            inf.com_cur = s.center_of_mass[i];
            auto numClusters = NUMBER_OF_CLUSTERS(i);
            inf.inv_numClusters = 1.0f / (float)numClusters;
        }

        queue.enqueueReadBuffer(d_q, CL_TRUE, 0, 3 * 4 * sizeof(float), s.predicted_orientation.data());

        for (unsigned i = 0; i < particle_count(s); i++) {
            float const stiffness = 1;
            auto const R = s.predicted_orientation[i];

            auto& inf = correction_infos[i];

            // Rotate the bind pose position relative to the CoM
            auto pos_bind_rot = R * inf.pos_bind;
            // Our goal position
            auto goal = inf.com_cur + pos_bind_rot;
            // Number of clusters this particle is a member of
            auto correction = (goal - s.predicted_position[i]) * stiffness;
            // The correction must be divided by the number of clusters this particle is a member of
            s.predicted_position[i] += inf.inv_numClusters * correction;
            s.goal_position[i] = goal;
        }

        END_BENCHMARK();
        PRINT_BENCHMARK_RESULT();
    }
};

static std::optional<cl::Program> from_file_load_program(
    char const* pszPath,
    cl::Context& ctx,
    std::vector<cl::Device> const& devices
) {
    Vector<std::string> chunks;

    FILE* f = fopen(pszPath, "rb");
    if (f != NULL) {
        std::string chunk("");
        while (!feof(f)) {
            chunk.resize(1024);
            auto res = fread(chunk.data(), 1, 1024, f);
            if (res > 0) {
                chunk.resize(res);
                chunks.push_back(std::move(chunk));
            }
        }
        fclose(f);
    } else {
        // Couldn't open file, return empty
        fprintf(stderr, "sb: couldn't open OpenCL source file '%s'\n", pszPath);
        return std::nullopt;
    }

    if (chunks.size() > 0) {
        cl::Program::Sources sources;
        for (auto& chunk : chunks) {
            sources.push_back({ chunk.c_str(), chunk.length() });
        }

        auto program = cl::Program(ctx, sources);
        try {
            program.build();
            return program;
        } catch (cl::Error& e) {
            if (e.err() == CL_BUILD_PROGRAM_FAILURE) {
                for (auto& dev : devices) {
                    auto status = program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(dev);
                    if (status != CL_BUILD_ERROR) {
                        continue;
                    }

                    auto dev_name = dev.getInfo<CL_DEVICE_NAME>();
                    auto log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev);
                    printf(
                        "sb: CL program build failure\n\ton device '%s'\n\tfile: '%s'\nbuild log:\n%s\n\t=== END OF BUILD LOG ===\n",
                        dev_name.c_str(), pszPath, log.c_str()
                    );
                }
            }

            return std::nullopt;
        }
    } else {
        // Empty file, return empty
        fprintf(stderr, "sb: OpenCL source file '%s' was empty\n", pszPath);
        return std::nullopt;
    }
}

sb::Unique_Ptr<ICompute_Backend> Make_CL_Backend() {
    try {
        Vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        Vector<cl::Device> devices;

        for (auto& platform : platforms) {
            Vector<cl::Device> devices_buffer;
            try {
                platform.getDevices(CL_DEVICE_TYPE_GPU, &devices_buffer);
                devices.insert(devices.end(), devices_buffer.begin(), devices_buffer.end());
            } catch (std::exception&) {
            }
        }

        if (devices.size() != 0) {
            printf("sb: OpenCL devices available (count=%zu):\n", devices.size());
            for (auto& device : devices) {
                std::string name;
                device.getInfo(CL_DEVICE_NAME, &name);
                printf("- %s\n", name.c_str());
                device.getInfo(CL_DEVICE_VENDOR, &name);
                printf("\tVendor: %s\n", name.c_str());
            }
            printf("\n");

            cl::Context ctx(devices);
            auto program = from_file_load_program("shape_matching.cl", ctx, devices);
            if (program) {
                return std::make_unique<Compute_CL>(std::move(ctx), std::move(*program));
            }
        }
    } catch(std::exception& e) {
        printf("sb: ex: %s\n", e.what());
    }

    fprintf(stderr, "sb: can't make CL compute backend\n");
    return NULL;
}
