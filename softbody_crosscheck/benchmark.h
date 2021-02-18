// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: benchmark declaration
//

#pragma once

#include <softbody.h>
#include <raymarching.h>

class Sample_Point : public sb::sdf::ast::Sample_Point {
public:
    glm::vec3 evaluate() override {
        return m_v;
    }

    void set_value(glm::vec3 const& v) override {
        m_v = v;
    }

    void visit(sb::sdf::ast::Visitor* v) const override { v->visit(*this); }

private:
    glm::vec3 m_v{};
};

class Test_Collider_Radius : public sb::sdf::ast::Float_Constant {
public:
    float evaluate() override { return 8.0f; }

    void value(float* out_array) const noexcept override {
        out_array[0] = 8.0f;
    }

    void set_value(float const* value) noexcept override {}

    void visit(sb::sdf::ast::Visitor* v) const override { v->visit(*this); }
};

class Test_Collider : public sb::sdf::ast::Primitive {
public:
    size_t parameter_count() const override {
        return 2;
    }

    void parameters(size_t count, Node const** out_arr) const override {
        if(count > 0) {
            out_arr[0] = &m_sp;
            if(count > 1) {
                out_arr[1] = &m_radius;
            }
        }
    }

    sb::sdf::ast::Primitive::Kind kind() const noexcept override {
        return sb::sdf::ast::Primitive::Kind::SPHERE;
    }

    void visit(sb::sdf::ast::Visitor* v) const override { v->visit(*this); }

    float evaluate() override {
        return sdf::sphere(m_radius.evaluate(), m_sp.evaluate());
    }

    Sample_Point m_sp;
    Test_Collider_Radius m_radius;
};

class Benchmark {
public:
    void run(float total_time);
    static Benchmark make_benchmark(sb::Compute_Preference backend, int dim);

private:
    Benchmark(sb::Unique_Ptr<sb::ISoftbody_Simulation>&& sim)
        : sim(std::move(sim)) {
        sb::ISoftbody_Simulation::Collider_Handle hc;
        this->sim->add_collider(hc, &coll, &coll.m_sp);
    }
    
private:
    sb::Unique_Ptr<sb::ISoftbody_Simulation> sim;
    Test_Collider coll;
};
