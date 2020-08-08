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
#include <glm/gtx/matrix_operation.hpp>
#include <glm/gtc/type_ptr.hpp>

#define CL_HPP_ENABLE_EXCEPTIONS 1
#include <CL/cl2.hpp>

#define NUMBER_OF_CLUSTERS(idx) (s.edges[(idx)].size() + 1)

class calc_mass;

class Compute_CL : public ICompute_Backend {
public:
    Compute_CL(cl::Context&& _ctx, cl::Device&& _dev, cl::Program&& _program)
        : ctx(std::move(_ctx)), dev(std::move(_dev)), program(std::move(_program)),
        k_test_mat_mul(program, "mat_mul_main"),
        k_test_rotation_extraction(program, "mueller_rotation_extraction"),
        k_calculate_particle_masses(program, "calculate_particle_masses"),
        k_do_shape_matching(program, "do_shape_matching")
    {
        queue = cl::CommandQueue(ctx);

        assert(sanity_check_mat_mul());
        assert(sanity_check_mueller_rotation_extraction());

        auto local_mem_size = k_do_shape_matching.getKernel().getWorkGroupInfo<CL_KERNEL_LOCAL_MEM_SIZE>(dev);
        auto private_mem_size = k_do_shape_matching.getKernel().getWorkGroupInfo<CL_KERNEL_PRIVATE_MEM_SIZE>(dev);
        auto preferred_work_group_size_multiple = k_do_shape_matching.getKernel().getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(dev);

        printf("sb: do_shape_matching kernel details:\n");
        printf("\tLocal memory size: %llu\n", local_mem_size);
        printf("\tPrivate memory size: %llu\n", private_mem_size);
        printf("\tPreferred work-group size multiple: %zu\n\n", preferred_work_group_size_multiple);
    }
private:
    cl::Context ctx;
    cl::Device dev;
    cl::Program program;
    cl::CommandQueue queue;

    using Test_Mat_Mul = cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer>;
    using Test_Rotation_Extraction = cl::KernelFunctor<cl::Buffer, cl::Buffer>;
    using Calculate_Particle_Masses = cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer>;
    using Do_Shape_Matching = cl::KernelFunctor<cl::Buffer, cl::Buffer, unsigned, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer >;

    Vector<float> particle_masses;
    cl::Buffer d_masses, d_predicted_orientations, d_sizes, d_predicted_positions;
    cl::Buffer d_bind_pose, d_centers_of_masses, d_bind_pose_centers_of_masses;
    cl::Buffer d_bind_pose_inverse_bind_pose;
    cl::Buffer d_adjacency;
    cl::Buffer d_out;

    Test_Mat_Mul k_test_mat_mul;
    Test_Rotation_Extraction k_test_rotation_extraction;
    Calculate_Particle_Masses k_calculate_particle_masses;
    Do_Shape_Matching k_do_shape_matching;

    float mass_of_particle(System_State const& s, unsigned i) const {
        assert(particle_masses.size() == particle_count(s));
        return particle_masses[i];
    }

    size_t particle_count(System_State const& s) const {
        return s.position.size();
    }

#define SIZE_N_VEC1(N) ((N) *  1 * sizeof(float))
#define SIZE_N_VEC4(N) ((N) *  4 * sizeof(float))
#define SIZE_N_MAT4(N) ((N) * 16 * sizeof(float))

    void begin_new_frame(System_State const& sim) override {
        auto const N = particle_count(sim);

        d_masses = cl::Buffer(ctx, CL_MEM_READ_WRITE, SIZE_N_VEC1(N));
        d_predicted_orientations = cl::Buffer(ctx, CL_MEM_READ_WRITE, SIZE_N_VEC4(N));
        d_sizes = cl::Buffer(ctx, CL_MEM_READ_ONLY, SIZE_N_VEC4(N));
        d_predicted_positions = cl::Buffer(ctx, CL_MEM_READ_ONLY, SIZE_N_VEC4(N));
        d_bind_pose = cl::Buffer(ctx, CL_MEM_READ_ONLY, SIZE_N_VEC4(N));
        d_centers_of_masses = cl::Buffer(ctx, CL_MEM_READ_ONLY, SIZE_N_VEC4(N));
        d_bind_pose_centers_of_masses = cl::Buffer(ctx, CL_MEM_READ_ONLY, SIZE_N_VEC4(N));
        d_bind_pose_inverse_bind_pose = cl::Buffer(ctx, CL_MEM_READ_ONLY, SIZE_N_MAT4(N));
        d_out = cl::Buffer(ctx, CL_MEM_READ_WRITE, SIZE_N_VEC4(N));

        particle_masses = calculate_particle_masses(sim);
    }

    Vector<float> calculate_particle_masses(System_State const& s) {
        auto N = particle_count(s);
        Vector<float> h_masses;
        Vector<Vec4> h_sizes;

        h_masses.resize(N);
        h_sizes.resize(N);

        for (auto i = 0ull; i < N; i++) {
            h_sizes[i] = Vec4(s.size[i], 0);
        }

        auto d_densities = cl::Buffer(ctx, CL_MEM_READ_ONLY, SIZE_N_VEC1(N));

        queue.enqueueWriteBuffer(d_densities, CL_FALSE, 0, SIZE_N_VEC1(N), s.density.data());
        queue.enqueueWriteBuffer(d_sizes, CL_FALSE, 0, SIZE_N_VEC4(N), h_sizes.data());

        k_calculate_particle_masses(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(N), cl::NDRange(1)), d_sizes, d_densities, d_masses);

        queue.enqueueReadBuffer(d_masses, true, 0, SIZE_N_VEC1(N), h_masses.data());

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

        queue.enqueueWriteBuffer(d_A, CL_FALSE, 0, 3 * 16 * sizeof(float), matrices);
        queue.enqueueWriteBuffer(d_q, CL_FALSE, 0, 3 *  4 * sizeof(float), approx);

        k_test_rotation_extraction(cl::EnqueueArgs(queue, cl::NDRange(3)), d_A, d_q);

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

        queue.enqueueWriteBuffer(d_lhs, CL_FALSE, 0, 16 * sizeof(float), glm::value_ptr(lhs));
        queue.enqueueWriteBuffer(d_rhs, CL_FALSE, 0, 16 * sizeof(float), glm::value_ptr(rhs));

        k_test_mat_mul(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(1), cl::NDRange(1)), d_out, d_lhs, d_rhs);

        queue.enqueueReadBuffer(d_out, CL_TRUE, 0, 16 * sizeof(float), glm::value_ptr(out));

        auto diff = matrix_difference(expected, out);

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

    // Generate the adjacency table
    // This table describes the connections between the particles.
    // It has N rows (where N is the particle count), each row has the neighbor count
    // in the first column, which is then followed by the corresponding amount of
    // particle indices.
    // If the particle with the most amount of neighbors has M neighbors, then each
    // row has M+1 columns.
    sb::Unique_Ptr<unsigned[]> make_adjacency_table(System_State const& s, unsigned* adjacency_stride, unsigned* adjacency_size) {
        auto const N = particle_count(s);
        unsigned columns = 0;
        for (unsigned i = 0; i < N; i++) {
            auto c = s.edges.at(i).size();
            if (c > columns) {
                columns = c;
            }
        }

        // Add one more slot for the neighbor count
        columns = columns + 1;

        auto ret = std::make_unique<unsigned[]>(N * columns);
        *adjacency_stride = columns;
        *adjacency_size = N * columns * sizeof(unsigned);

        for (unsigned i = 0; i < N; i++) {
            auto base = i * columns;
            auto& neighbors = s.edges.at(i);
            auto const M = neighbors.size();
            ret[base] = M;
            base++;
            for (unsigned j = 0; j < M; j++) {
                ret[base] = neighbors[j];
                base++;
            }
        }

        return ret;
    }

    // Produces a number that describes how different are these two matrices.
    // A value of (near) zero means that they're (almost) equal.
    // A non-zero value has no meaning other than that the two matrices are different.
    float matrix_difference(glm::mat4 const& lhs, glm::mat4 const& rhs) {
        auto sum_mat = [](glm::mat4 const& m) {
            float acc = 0;

            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    acc += glm::abs(m[i][j]);
                }
            }

            return acc;
        };

        return sum_mat(lhs - rhs);
    }

    void do_one_iteration_of_shape_matching_constraint_resolution(
        System_State& s,
        float phdt
    ) override {
        DECLARE_BENCHMARK_BLOCK();
        BEGIN_BENCHMARK();

        auto const N = particle_count(s);
        std::vector<Vec4> h_centers_of_masses;
        h_centers_of_masses.reserve(N);

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

            h_centers_of_masses.push_back(Vec4(com_cur, 0));
        }

        unsigned adjacency_stride, adjacency_size;
        auto adjacency = make_adjacency_table(s, &adjacency_stride, &adjacency_size);

        // Convert Vec3's into Vec4's before we upload them to the GPU
        // TODO(danielm): this has a significant performance hit.
        // Convert all Vec3 type values into Vec4's in System_State

        Vector<Vec4> h_predicted_positions, h_bind_pose, h_bind_pose_centers_of_masses;
        Vector<glm::mat4> h_bind_pose_inverse_bind_pose;
        Vector<Quat> h_out;

        h_predicted_positions.resize(N);
        h_bind_pose.resize(N);
        h_bind_pose_centers_of_masses.resize(N);
        h_bind_pose_inverse_bind_pose.resize(N);

        d_adjacency = cl::Buffer(ctx, CL_MEM_READ_ONLY, adjacency_size);

        queue.enqueueWriteBuffer(d_predicted_orientations, CL_FALSE, 0, SIZE_N_VEC4(N), s.predicted_orientation.data());
        queue.enqueueWriteBuffer(d_adjacency, CL_FALSE, 0, adjacency_size, adjacency.get());
        queue.enqueueWriteBuffer(d_centers_of_masses, CL_FALSE, 0, SIZE_N_VEC4(N), h_centers_of_masses.data());
        for (unsigned i = 0; i < N; i++) h_predicted_positions[i] = Vec4(s.predicted_position[i], 0);
        queue.enqueueWriteBuffer(d_predicted_positions, CL_FALSE, 0, SIZE_N_VEC4(N), h_predicted_positions.data());
        for (unsigned i = 0; i < N; i++) h_bind_pose[i] = Vec4(s.bind_pose[i], 0);
        queue.enqueueWriteBuffer(d_bind_pose, CL_FALSE, 0, SIZE_N_VEC4(N), h_bind_pose.data());
        for (unsigned i = 0; i < N; i++) h_bind_pose_centers_of_masses[i] = Vec4(s.bind_pose_center_of_mass[i], 0);
        queue.enqueueWriteBuffer(d_bind_pose_centers_of_masses, CL_FALSE, 0, SIZE_N_VEC4(N), h_bind_pose_centers_of_masses.data());
        for (unsigned i = 0; i < N; i++) h_bind_pose_inverse_bind_pose[i] = glm::mat4(s.bind_pose_inverse_bind_pose[i]);
        queue.enqueueWriteBuffer(d_bind_pose_inverse_bind_pose, CL_FALSE, 0, SIZE_N_MAT4(N), h_bind_pose_inverse_bind_pose.data());

        k_do_shape_matching(cl::EnqueueArgs(queue, cl::NDRange(N)),
            d_out,
            d_adjacency, adjacency_stride,
            d_masses,
            d_predicted_orientations, d_sizes, d_predicted_positions, d_bind_pose,
            d_centers_of_masses, d_bind_pose_centers_of_masses,
            d_bind_pose_inverse_bind_pose
        );

#if CALC_A_I_PARANOID
        // Check if the matrix calculated by the GPU kernel and the matrix
        // calculated by the reference implementations match.
        auto calc_A_i = [&](unsigned i) -> glm::mat4 {
            auto m_i = mass_of_particle(s, i);
            auto diag = glm::diagonal4x4(Vec4(s.size[i], 0) * Vec4(s.size[i], 0));
            auto orient = glm::mat4(s.predicted_orientation[i]);
            auto A_i = 1.0f / 5.0f * diag * orient;

            auto t0 = A_i + glm::outerProduct(h_predicted_positions[i], h_bind_pose[i]);

            auto t1 = (t0 - glm::outerProduct(h_centers_of_masses[i], h_bind_pose_centers_of_masses[i]));

            return m_i * t1;
        };


        Vector<glm::mat4> h_test_out;
        Vector<glm::mat4> expected;
        h_test_out.resize(N);
        expected.resize(N);
        auto d_test_out = cl::Buffer(ctx, CL_MEM_WRITE_ONLY, SIZE_N_MAT4(N));
        auto test_kernel = cl::KernelFunctor<cl::Buffer, unsigned, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer>(program, "test_calculate_A_i");
        test_kernel(cl::EnqueueArgs(queue, cl::NDRange(1)), d_test_out, N, d_masses, d_predicted_orientations, d_sizes, d_predicted_positions, d_bind_pose, d_centers_of_masses, d_bind_pose_centers_of_masses);

        for (unsigned i = 0; i < N; i++) {
            expected[i] = calc_A_i(i);
        }

        queue.enqueueReadBuffer(d_test_out, CL_TRUE, 0, SIZE_N_MAT4(N), h_test_out.data());

        for (unsigned i = 0; i < N; i++) {
            auto diff = matrix_difference(expected[i], h_test_out[i]);
            if (diff > glm::epsilon<float>()) {
                assert(!"Calculated A_i matrix doesn't match expected value!");
            }
        }
#endif /* CALC_A_I_PARANOID */

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

        h_out.resize(N);

        queue.enqueueReadBuffer(d_out, CL_TRUE, 0, SIZE_N_VEC4(N), h_out.data());

        for (unsigned i = 0; i < particle_count(s); i++) {
            float const stiffness = 1;
            auto const R = h_out[i];

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
            s.predicted_orientation[i] = R;
        }

        END_BENCHMARK();
        PRINT_BENCHMARK_RESULT_MASKED(0xFF);
    }
};

static std::optional<cl::Program> from_file_load_program(
    char const* pszPath,
    cl::Context& ctx,
    cl::Device& dev 
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

            auto dev_name = dev.getInfo<CL_DEVICE_NAME>();
            auto log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev);
            if (!log.empty()) {
                printf(
                    "sb: CL program built with warnings:\n\ton device '%s'\n\tfile: '%s'\nbuild log:\n%s\n\t=== END OF BUILD LOG ===\n",
                    dev_name.c_str(), pszPath, log.c_str()
                );
            }

            return program;
        } catch (cl::Error& e) {
            if (e.err() == CL_BUILD_PROGRAM_FAILURE || e.err() == -9999) {
                auto status = program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(dev);
                if (status != CL_BUILD_ERROR) {
                    return std::nullopt;
                }

                auto dev_name = dev.getInfo<CL_DEVICE_NAME>();
                auto log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev);
                printf(
                    "sb: CL program build failure\n\ton device '%s'\n\tfile: '%s'\nbuild log:\n%s\n\t=== END OF BUILD LOG ===\n",
                    dev_name.c_str(), pszPath, log.c_str()
                );
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
        std::optional<cl::Device> selected_device;
        Vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);

        for (auto& platform : platforms) {
            Vector<cl::Device> devices_buffer;
            try {
                platform.getDevices(CL_DEVICE_TYPE_GPU, &devices_buffer);
                if (devices_buffer.size() > 0) {
                    selected_device = devices_buffer[0];
                    break;
                }
            } catch (cl::Error& e1) {
                printf("sb: couldn't enumerate OpenCL GPU devices: [%d] %s\n", e1.err(), e1.what());
                try {
                    platform.getDevices(CL_DEVICE_TYPE_CPU, &devices_buffer);
                    if (devices_buffer.size() > 0) {
                        selected_device = devices_buffer[0];
                        break;
                    }
                } catch (cl::Error& e2) {
                    printf("sb: couldn't enumerate OpenCL CPU devices: [%d] %s\n", e2.err(), e2.what());
                }
            }
        }

        if (selected_device) {
            printf("sb: OpenCL device selected:\n");
            std::string name;
            selected_device->getInfo(CL_DEVICE_NAME, &name);
            printf("- %s\n", name.c_str());
            selected_device->getInfo(CL_DEVICE_VENDOR, &name);
            printf("\tVendor: %s\n", name.c_str());
            printf("\tCompute units: %d\n", selected_device->getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>());
            printf("\tGlobal memory: %llu\n", selected_device->getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>());
            printf("\tConstant memory: %llu\n", selected_device->getInfo<CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE>());
            printf("\n");

            cl::Context ctx(*selected_device);
            auto program = from_file_load_program("shape_matching.cl", ctx, *selected_device);
            if (program) {
                return std::make_unique<Compute_CL>(std::move(ctx), std::move(*selected_device), std::move(*program));
            }
        }
    } catch(cl::Error& e) {
        printf("sb: couldn't create CL backend: [%d] %s\n", e.err(), e.what());
    }

    fprintf(stderr, "sb: can't make CL compute backend\n");
    return NULL;
}
