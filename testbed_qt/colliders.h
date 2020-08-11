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
#include <QDebug>
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
    Q_OBJECT;
public:
    virtual ~Base_Collider_Widget() {}

    Base_Collider_Widget() {
        frame.setupUi(this);

        connect(frame.btnDelete, &QPushButton::released, this, &Base_Collider_Widget::remove_self);
    }

    virtual void added_to_simulation(sb::ISoftbody_Simulation* sim, unsigned handle) override {
        sim_handle = { sim, handle };
    }

    void set_remover(std::function<void(Base_Collider_Widget*, unsigned)> const& f) {
        remover = f;
    }

protected:
    QWidget* container() { return frame.collider; }

private slots:
    void remove_self() {
        if (sim_handle) {
            assert(remover);
            remover(this, std::get<1>(*sim_handle));
        } else {
            qDebug() << "collider widget has empty sim_handle";
        }
    }

private:
    Ui::Collider_Frame frame;
    std::optional<std::tuple<sb::ISoftbody_Simulation*, unsigned>> sim_handle;
    std::function<void(Base_Collider_Widget*, unsigned)> remover;
};

class Sphere_Collider_Widget : public Base_Collider_Widget {
    Q_OBJECT;
public:
    virtual ~Sphere_Collider_Widget() {}

    Sphere_Collider_Widget()
        : radius(1.0f) {
        form.setupUi(container());
        connect(form.sbRadius, (void (QDoubleSpinBox::*)(double))&QDoubleSpinBox::valueChanged, this, &Sphere_Collider_Widget::set_radius);
    }

    virtual float evaluate(glm::vec3 const& sp) override {
        return sdf::sphere(radius, sp);
    }

    Q_PROPERTY(float radius READ get_radius WRITE set_radius);

    float get_radius() const { return radius; }
    void set_radius(float v) { radius = v; }

protected:

private:
    Ui::Collider_Sphere form;
    float radius;
};
