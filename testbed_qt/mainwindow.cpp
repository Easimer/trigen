// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: main window implementation
//

#include "common.h"
#include "mainwindow.h"
#include <QVariant>
#include "softbody_renderer.h"

#define BIND_DATA_DBLSPINBOX(field, elem)                                                                   \
    connect(                                                                                                \
        &sim_cfg, &Simulation_Config::field##_changed,                                                    \
        sim_config.elem, &QDoubleSpinBox::setValue                                                          \
    );                                                                                                      \
    connect(                                                                                                \
        sim_config.elem, static_cast<void (QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged),      \
        &sim_cfg, [&](float v) { sim_cfg.setProperty(#field, v); }                                          \
    );                                                                                                      \

#define BIND_DATA_SPINBOX(field, elem)                                                      \
    connect(                                                                                \
        &sim_cfg, &Simulation_Config::field##_changed,                                    \
        sim_config.elem, &QSpinBox::setValue                                                \
    );                                                                                      \
    connect(                                                                                \
        sim_config.elem, static_cast<void (QSpinBox::*)(int )>(&QSpinBox::valueChanged),    \
        &sim_cfg, [&](int v) { sim_cfg.setProperty(#field, (unsigned)v); }                  \
    );                                                                                      \

#define BIND_DATA_COMBOBOX(field, elem)                                                 \
    connect(                                                                            \
        &sim_cfg, &Simulation_Config::field##_changed,                                \
        [&](Simulation_Config::Extension ext) { \
sim_config.elem->setCurrentText(                   \
    QVariant::fromValue(ext).toString() \
);  \
        }   \
    );                                                                                  \
    connect(                                                                            \
        sim_config.elem, (void (QComboBox::*)(QString const&))&QComboBox::currentIndexChanged,     \
        &sim_cfg, [&](QString const& v) { sim_cfg.setProperty(#field, v); }                        \
    );                                                                                  \

#define BIND_DATA_VEC3_COMPONENT(obj_field, field, sig, set)                                            \
    connect(                                                                                            \
        &sim_cfg.obj_field, &QVec3::sig,                                                                \
        sim_config.field, &QDoubleSpinBox::setValue);                                                   \
    connect(                                                                                            \
        sim_config.field, static_cast<void (QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged), \
        &sim_cfg.seed_position, &QVec3::set);                                                           \

#define BIND_DATA_VEC3(vec, fieldx, fieldy, fieldz)             \
    BIND_DATA_VEC3_COMPONENT(vec, fieldx, x_changed, set_x);    \
    BIND_DATA_VEC3_COMPONENT(vec, fieldy, y_changed, set_y);    \
    BIND_DATA_VEC3_COMPONENT(vec, fieldz, z_changed, set_z);    \

MainWindow::MainWindow(QWidget* parent) :
    QMainWindow(parent),
    render_timer(this),
    sim_cfg(),
    splitter(std::make_unique<QSplitter>(this)),
    render_params() {
    splitter->setOrientation(Qt::Horizontal);

    auto tabs = new QTabWidget(this);
    auto gl_viewport = new GLViewport(this);

    sim_control_widget = std::make_unique<QWidget>(this);
    sim_control.setupUi(sim_control_widget.get());
    tabs->addTab(sim_control_widget.get(), "Simulation control");

    sim_config_widget = std::make_unique<QWidget>(this);
    sim_config.setupUi(sim_config_widget.get());
    tabs->addTab(sim_config_widget.get(), "Simulation parameters");

    // NOTE(danielm): ownership passed to the QSplitter
    splitter->addWidget(tabs);
    splitter->addWidget(gl_viewport);

    setCentralWidget(splitter.get());

    viewport = gl_viewport;
    viewport->set_render_queue_filler([this](gfx::Render_Queue* rq) { render_world(rq); });

    connect(&render_timer, SIGNAL(timeout()), viewport, SLOT(update()));
    render_timer.start(13);

    connect(sim_control.sliderSimSpeed, &QSlider::valueChanged, [&](int value) {
        char buf[64];
        auto speed = value / 4.0f;
        auto res = snprintf(buf, 63, "%.2fx", speed);
        buf[res] = 0;
        sim_control.lblSpeedValue->setText((char const*)buf);
        this->sim_speed = speed;
    });

    connect(sim_control.btnStart, SIGNAL(released()), this, SLOT(start_simulation()));
    connect(sim_control.btnStop, SIGNAL(released()), this, SLOT(stop_simulation()));
    connect(sim_control.btnReset, SIGNAL(released()), this, SLOT(reset_simulation()));

    extensions.insert("#0 None", sb::Extension::None);
    extensions.insert("#1 Rope demo", sb::Extension::Debug_Rope);
    extensions.insert("#2 Cloth demo", sb::Extension::Debug_Cloth);
    extensions.insert("#3 Plant sim", sb::Extension::Plant_Simulation);

    sim_config.cbExtension->addItems(QStringList(extensions.keys()));
    sim_config.cbExtension->setCurrentIndex(0);

    BIND_DATA_SPINBOX(particle_count_limit, sbParticleCountLimit);
    BIND_DATA_DBLSPINBOX(density, sbDensity);
    BIND_DATA_DBLSPINBOX(attachment_strength, sbAttachmentStrength);
    BIND_DATA_DBLSPINBOX(surface_adaption_strength, sbAdaptionStrength);
    BIND_DATA_DBLSPINBOX(stiffness, sbStiffness);
    BIND_DATA_DBLSPINBOX(aging_rate, sbAgingRate);
    BIND_DATA_DBLSPINBOX(phototropism_response_strength, sbPhototropismResponseStrength);
    BIND_DATA_DBLSPINBOX(branching_probability, sbBranchingProbability);
    BIND_DATA_DBLSPINBOX(branch_angle_variance, sbBranchAngleVariance);
    BIND_DATA_VEC3(seed_position, sbOriginX, sbOriginY, sbOriginZ);

    connect(sim_config.cbExtension, &QComboBox::currentTextChanged, this, &MainWindow::on_extension_changed);

    // Default sim config
    auto extension = sim_cfg.property("ext");
    sim_cfg.setProperty("ext", "Plant_Simulation");
    extension = sim_cfg.property("ext");
    sim_cfg.seed_position = Vec3(1, 2, 3);
    sim_cfg.setProperty("density", 1.0f);
    sim_cfg.setProperty("attachment_strength", 1.0f);
    sim_cfg.setProperty("surface_adaption_strength", 1.0f);
    sim_cfg.setProperty("stiffness", 0.2f);
    sim_cfg.setProperty("aging_rate", 0.1f);
    sim_cfg.setProperty("phototropism_response_strength", 1.0f);
    sim_cfg.setProperty("branching_probability", 0.25f);
    sim_cfg.setProperty("branch_angle_variance", glm::pi<float>());
    sim_cfg.setProperty("particle_count_limit", 128u);
}

void MainWindow::start_simulation() {
    if (!conn_sim_step.has_value()) {
        if (!simulation) {
            reset_simulation();
        }
        conn_sim_step = connect(&render_timer, SIGNAL(timeout()), this, SLOT(step_simulation()));
    }
}

void MainWindow::render_world(gfx::Render_Queue* rq) {
    assert(rq != NULL);
    render_softbody_simulation(rq, simulation.get(), render_params);
}

void MainWindow::stop_simulation() {
    if (conn_sim_step.has_value()) {
        disconnect(*conn_sim_step);
        conn_sim_step.reset();
    }
}

void MainWindow::reset_simulation() {
    simulation = sb::create_simulation(sim_cfg);
}

void MainWindow::step_simulation() {
    if (simulation) {
        simulation->step(sim_speed * render_timer.interval() / 1000.0f);
    }
}

void MainWindow::on_extension_changed(QString const& k) {
    sim_cfg.ext = extensions.value(sim_config.cbExtension->currentText(), sb::Extension::None);
}
