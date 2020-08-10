// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: main window implementation
//

#include "common.h"
#include "mainwindow.h"
#include <QVariant>

#define BIND_DATA_DBLSPINBOX(field, elem)                                                                   \
    connect(                                                                                                \
        &sim_cfg, &Simulation_Config::##field##_changed,                                                    \
        sim_config.elem, &QDoubleSpinBox::setValue                                                          \
    );                                                                                                      \
    connect(                                                                                                \
        sim_config.elem, static_cast<void (QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged),      \
        &sim_cfg, [&](float v) { sim_cfg.setProperty(#field, v); }                                          \
    );                                                                                                      \

#define BIND_DATA_SPINBOX(field, elem)                                                      \
    connect(                                                                                \
        &sim_cfg, &Simulation_Config::##field##_changed,                                    \
        sim_config.elem, &QSpinBox::setValue                                                \
    );                                                                                      \
    connect(                                                                                \
        sim_config.elem, static_cast<void (QSpinBox::*)(int )>(&QSpinBox::valueChanged),    \
        &sim_cfg, [&](int v) { sim_cfg.setProperty(#field, (unsigned)v); }                  \
    );                                                                                      \

#define BIND_DATA_COMBOBOX(field, elem)                                                 \
    connect(                                                                            \
        &sim_cfg, &Simulation_Config::##field##_changed,                                \
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

// TODO(danielm): move these elsewhere
class Render_Grid : public gfx::IRender_Command {
private:
    virtual void execute(gfx::IRenderer* renderer) override {
        glm::vec3 lines[] = {
            glm::vec3(0, 0, 0),
            glm::vec3(1, 0, 0),
            glm::vec3(0, 0, 0),
            glm::vec3(0, 1, 0),
            glm::vec3(0, 0, 0),
            glm::vec3(0, 0, 1),
        };

        renderer->draw_lines(lines + 0, 1, Vec3(0, 0, 0), Vec3(.35, 0, 0), Vec3(1, 0, 0));
        renderer->draw_lines(lines + 2, 1, Vec3(0, 0, 0), Vec3(0, .35, 0), Vec3(0, 1, 0));
        renderer->draw_lines(lines + 4, 1, Vec3(0, 0, 0), Vec3(0, 0, .35), Vec3(0, 0, 1));

        // render grid
        Vec3 grid[80];
        for (int i = 0; i < 20; i++) {
            auto base = 4 * i;
            grid[base + 0] = Vec3(i - 10, 0, -10);
            grid[base + 1] = Vec3(i - 10, 0, +10);
            grid[base + 2] = Vec3(-10, 0, i - 10);
            grid[base + 3] = Vec3(+10, 0, i - 10);
        }

        renderer->draw_lines(grid, 40, Vec3(0, 0, 0), Vec3(0.4, 0.4, 0.4), Vec3(0.4, 0.4, 0.4));
    }
};

template<typename T, class ... Arg>
static T* allocate_command_and_initialize(gfx::Render_Queue* rq, Arg ... args) {
    auto cmd = rq->allocate<T>();
    new(cmd) T(args...);
    rq->push(cmd);
    return cmd;
}

static void rqfiller(gfx::Render_Queue* rq) {
    allocate_command_and_initialize<Render_Grid>(rq);
}

MainWindow::MainWindow(QWidget* parent) :
    QMainWindow(parent),
    render_timer(this),
    sim_cfg(),
    splitter(std::make_unique<QSplitter>(this)) {
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
    viewport->set_render_queue_filler(&rqfiller);

    connect(&render_timer, SIGNAL(timeout()), viewport, SLOT(update()));
    conn_sim_step = connect(&render_timer, SIGNAL(timeout()), this, SLOT(step_simulation()));
    render_timer.start(13);

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
        printf("Starting simulation\n");
        if (!simulation) {
            reset_simulation();
        }
        conn_sim_step = connect(&render_timer, SIGNAL(timeout()), this, SLOT(step_simulation()));
    }
}

void MainWindow::stop_simulation() {
    if (conn_sim_step.has_value()) {
        printf("Stopping simulation\n");
        disconnect(*conn_sim_step);
        conn_sim_step.reset();
    }
}

void MainWindow::reset_simulation() {
    printf("Resetting simulation\n");
    simulation = sb::create_simulation(sim_cfg);
}

void MainWindow::step_simulation() {
    if (simulation) {
        printf("Stepping simulation\n");
        simulation->step(render_timer.interval() / 1000.0f);
    }
}

void MainWindow::on_extension_changed(QString const& k) {
    sim_cfg.ext = extensions.value(sim_config.cbExtension->currentText(), sb::Extension::None);
}
