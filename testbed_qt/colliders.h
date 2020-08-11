// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: collider UI
//

#pragma once

#include <optional>
#include <utility>
#include <memory>
#include <glm/vec3.hpp>
#include "softbody.h"
#include <QWidget>
#include "ui/ui_collider_frame.h"
#include "ui/ui_collider_sphere.h"

class Base_Collider {
public:
    virtual ~Base_Collider() {}
    virtual float evaluate(glm::vec3 const& sp) = 0;
    virtual void added_to_simulation(sb::ISoftbody_Simulation* sim, unsigned handle) = 0;
};

class Collider_Proxy {
public:
    using Collider_Ref = Base_Collider*;

    Collider_Proxy(Collider_Ref collider)
    : collider(collider) {
        assert(collider != NULL);
    }

    float operator()(glm::vec3 const& sp) {
        assert(collider != NULL);
        return collider->evaluate(sp);
    }
private:
    Collider_Ref collider;
};

inline void add_collider(sb::ISoftbody_Simulation* sim, std::unique_ptr<Base_Collider>& coll) {
    if (coll != NULL) {
        Collider_Proxy p(coll.get());
        auto handle = sim->add_collider(p);
        coll->added_to_simulation(sim, handle);
    }
}

class Base_Collider_Widget : public QWidget, public Base_Collider {
public:
    Base_Collider_Widget() {
        frame.setupUi(this);
    }

    virtual void added_to_simulation(sb::ISoftbody_Simulation* sim, unsigned handle) override {
        assert(!sim_handle.has_value());

        sim_handle = { sim, handle };
    }

protected:
    QWidget* container() { return frame.collider; }

private:
    Ui::Collider_Frame frame;
    std::optional<std::tuple<sb::ISoftbody_Simulation*, unsigned>> sim_handle;
};

class Sphere_Collider_Widget : public Base_Collider_Widget {
    Q_OBJECT;
public:
    Sphere_Collider_Widget() {
        form.setupUi(container());
    }

    virtual float evaluate(glm::vec3 const& sp) override {
        return sdf::sphere(radius, sp);
    }

    Q_PROPERTY(float radius READ getRadius WRITE setRadius);

    float getRadius() const { return radius; }
    void setRadius(float v) { radius = v; }

private:
    Ui::Collider_Sphere form;
    float radius;
};
