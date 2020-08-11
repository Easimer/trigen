// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: main window declaration
//

#pragma once

#include "common.h"
#include <list>
#include <QMainWindow>
#include <QTimer>
#include <QSplitter>
#include <QToolBar>
#include "glviewport.h"
#include "r_queue.h"
#include "ui_sim_control.h"
#include "ui_sim_config.h"

#include "softbody.h"
#include "softbody_renderer.h"
#include "colliders.h"

/*
 * Wraps a glm::vec3 in a Qt object by reference.
 */
class QVec3 : public QObject {
    Q_OBJECT;

    Q_PROPERTY(float x READ get_x WRITE set_x NOTIFY x_changed);
    Q_PROPERTY(float y READ get_y WRITE set_y NOTIFY y_changed);
    Q_PROPERTY(float z READ get_z WRITE set_z NOTIFY z_changed);
public:
    QVec3(glm::vec3& ref) : ref(ref) {}

    glm::vec3& operator=(glm::vec3 const& other) {
        set_x(other.x);
        set_y(other.y);
        set_z(other.z);

        return ref;
    }

    void set_x(float v) {
        if (ref.x != v) {
            ref.x = v;
            emit x_changed(v);
        }
    }

    void set_y(float v) {
        if (ref.y != v) {
            ref.y = v;
            emit y_changed(v);
        }
    }

    void set_z(float v) {
        if (ref.z != v) {
            ref.z = v;
            emit z_changed(v);
        }
    }

    float get_x() const { return ref.x; }
    float get_y() const { return ref.y; }
    float get_z() const { return ref.z; }

signals:
    void x_changed(float v);
    void y_changed(float v);
    void z_changed(float v);
private:
    glm::vec3& ref;
};

Q_DECLARE_METATYPE(sb::Extension);

/*
 * Wraps the sb::Config structure in a Qt object
 */
class Simulation_Config : public QObject, public sb::Config {
    Q_OBJECT;
public:
    Q_PROPERTY(float density MEMBER density NOTIFY density_changed);
    Q_PROPERTY(float attachment_strength MEMBER attachment_strength NOTIFY attachment_strength_changed);
    Q_PROPERTY(float surface_adaption_strength MEMBER surface_adaption_strength NOTIFY surface_adaption_strength_changed);
    Q_PROPERTY(float stiffness MEMBER stiffness NOTIFY stiffness_changed);
    Q_PROPERTY(float aging_rate MEMBER aging_rate NOTIFY aging_rate_changed);
    Q_PROPERTY(float phototropism_response_strength MEMBER phototropism_response_strength NOTIFY phototropism_response_strength_changed);
    Q_PROPERTY(float branching_probability MEMBER branching_probability NOTIFY branching_probability_changed);
    Q_PROPERTY(float branch_angle_variance MEMBER branch_angle_variance NOTIFY branch_angle_variance_changed);
    Q_PROPERTY(unsigned particle_count_limit MEMBER particle_count_limit NOTIFY particle_count_limit_changed);
    Q_PROPERTY(sb::Extension ext MEMBER ext NOTIFY ext_changed);

    QVec3 seed_position = QVec3(sb::Config::seed_position);

signals:
    void density_changed(float value);
    void attachment_strength_changed(float value);
    void surface_adaption_strength_changed(float value);
    void stiffness_changed(float value);
    void aging_rate_changed(float value);
    void phototropism_response_strength_changed(float value);
    void branching_probability_changed(float value);
    void branch_angle_variance_changed(float value);
    void particle_count_limit_changed(unsigned value);
    void ext_changed(sb::Extension value);
};

template<typename UI>
class Ui_Widget {
public:
    Ui_Widget(QWidget* parent = nullptr)
        : ui(), widget(std::make_unique<QWidget>(parent)) {
        ui.setupUi(widget.get());
    }

    explicit operator QWidget* () {
        return widget.get();
    }

    UI* operator->() {
        return &ui;
    }
private:
    UI ui;
    Unique_Ptr<QWidget> widget;
};

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(QWidget *parent = nullptr);
    void set_render_queue_filler(Fun<void(gfx::Render_Queue* rq)> const& f) {
        viewport->set_render_queue_filler(f);
    }

    void render_world(gfx::Render_Queue* rq);

protected slots:
    void start_simulation();
    void stop_simulation();
    void reset_simulation();
    void step_simulation();
    void on_extension_changed(QString const& k);

private:
    void add_collider(Unique_Ptr<Base_Collider_Widget>&& widget);

    template<typename T>
    void add_collider() {
        auto w = std::make_unique<T>();
        add_collider(std::move(w));
    }

private:
    QTimer render_timer;
    Unique_Ptr<QSplitter> splitter;
    GLViewport* viewport; // owned by splitter

    Ui_Widget<Ui::Sim_Control> sim_control;
    float sim_speed = 1.0f;
    Optional<QMetaObject::Connection> conn_sim_step;

    Ui_Widget<Ui::Sim_Config> sim_config;

    sb::Unique_Ptr<sb::ISoftbody_Simulation> simulation;
    Simulation_Config sim_cfg;

    QMap<QString, sb::Extension> extensions;

    Softbody_Render_Parameters render_params;

    std::list<Unique_Ptr<Base_Collider_Widget>> collider_list;
    Unique_Ptr<QWidget> collider_list_widget;
    QVBoxLayout collider_list_layout;
    Unique_Ptr<QToolBar> collider_list_menubar;
};
