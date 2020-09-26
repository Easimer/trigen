// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: collider node graph
//

#pragma once

#include <memory>
#include <QWidget>
#include <glm/vec3.hpp>
#include <softbody.h>

class Collider_Builder_Widget {
public:
    virtual ~Collider_Builder_Widget() {}
    virtual QWidget* view() = 0;
    [[deprecated]]
    virtual float evaluate(glm::vec3 const& sample_point) = 0;

    virtual void get_ast(sb::sdf::ast::Expression<float>** expr, sb::sdf::ast::Sample_Point** sp) = 0;
};

std::unique_ptr<Collider_Builder_Widget> create_collider_builder_widget(QWidget* parent);
