// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: collider node graph
//

#pragma once

#include <memory>
#include <QWidget>
#include <glm/vec3.hpp>

class Collider_Builder_Widget {
public:
    virtual ~Collider_Builder_Widget() {}
    virtual QWidget* view() = 0;
    virtual float evaluate(glm::vec3 const& sample_point) = 0;
};

std::unique_ptr<Collider_Builder_Widget> create_collider_builder_widget(QWidget* parent);

class Softbody_Collider_Proxy {
public:
    Softbody_Collider_Proxy(std::unique_ptr<Collider_Builder_Widget>& cwb) : cwb(cwb.get()) {}

    float operator()(glm::vec3 const& sp) {
        return cwb->evaluate(sp);
    }
private:
    Collider_Builder_Widget* cwb;
};
