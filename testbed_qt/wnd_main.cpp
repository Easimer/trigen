// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: main window implementation
//

#include "common.h"
#include "wnd_main.h"
#include <QVariant>
#include <QMessageBox>
#include <QFileDialog>
#include <QThread>
#include "softbody_renderer.h"
#include "raymarching.h"
#include "colliders.h"
#include "wnd_meshgen.h"
#include <thread>
#include <objscan.h>

#define BIND_DATA_DBLSPINBOX(field, elem)                                                                   \
    connect(                                                                                                \
        &sim_cfg, &Simulation_Config::field##_changed,                                                      \
        sim_config->elem, &QDoubleSpinBox::setValue                                                         \
    );                                                                                                      \
    connect(                                                                                                \
        sim_config->elem, static_cast<void (QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged),     \
        &sim_cfg, [&](float v) { sim_cfg.setProperty(#field, v); }                                          \
    );                                                                                                      \

#define BIND_DATA_SPINBOX(field, elem)                                                      \
    connect(                                                                                \
        &sim_cfg, &Simulation_Config::field##_changed,                                      \
        sim_config->elem, &QSpinBox::setValue                                               \
    );                                                                                      \
    connect(                                                                                \
        sim_config->elem, static_cast<void (QSpinBox::*)(int )>(&QSpinBox::valueChanged),   \
        &sim_cfg, [&](int v) { sim_cfg.setProperty(#field, (unsigned)v); }                  \
    );                                                                                      \

#define BIND_DATA_COMBOBOX(field, elem)                 \
    connect(                                            \
        &sim_cfg, &Simulation_Config::field##_changed,  \
        [&](Simulation_Config::Extension ext) {         \
            sim_config.elem->setCurrentText(            \
                QVariant::fromValue(ext).toString()     \
            );                                          \
        }                                               \
    );                                                  \
    connect(                                                                                    \
        sim_config.elem, (void (QComboBox::*)(QString const&))&QComboBox::currentIndexChanged,  \
        &sim_cfg, [&](QString const& v) { sim_cfg.setProperty(#field, v); }                     \
    );                                                                                          \

#define BIND_DATA_VEC3_COMPONENT(obj_field, field, sig, set)                                             \
    connect(                                                                                             \
        &sim_cfg.obj_field, &QVec3::sig,                                                                 \
        sim_config->field, &QDoubleSpinBox::setValue);                                                   \
    connect(                                                                                             \
        sim_config->field, static_cast<void (QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged), \
        &sim_cfg.seed_position, &QVec3::set);                                                            \

#define BIND_DATA_VEC3(vec, fieldx, fieldy, fieldz)             \
    BIND_DATA_VEC3_COMPONENT(vec, fieldx, x_changed, set_x);    \
    BIND_DATA_VEC3_COMPONENT(vec, fieldy, y_changed, set_y);    \
    BIND_DATA_VEC3_COMPONENT(vec, fieldz, z_changed, set_z);    \

std::unique_ptr<sb::ISerializer> make_serializer(QString const& path);
std::unique_ptr<sb::IDeserializer> make_deserializer(QString const& path);

static void sb_msg_callback(sb::Debug_Message_Source src, sb::Debug_Message_Type type, sb::Debug_Message_Severity severity, char const* msg, void* user) {
    printf("sb: %s\n", msg);
}

Window_Main::Window_Main(QWidget* parent) :
    QMainWindow(parent),
    render_timer(this),
    sim_cfg(),
    splitter(std::make_unique<QSplitter>(this)),
    render_params(),
    sim_control(this),
    collider_builder(create_collider_builder_widget(this)) {
    splitter->setOrientation(Qt::Horizontal);

    auto tabs = new QTabWidget(this);
    auto gl_viewport = new GLViewport(this);

    tabs->addTab((QWidget*)sim_control, "Simulation control");
    tabs->addTab((QWidget*)sim_config, "Simulation parameters");
    tabs->addTab(collider_builder->view(), "Colliders");

    // NOTE(danielm): ownership passed to the QSplitter
    splitter->addWidget(tabs);
    splitter->addWidget(gl_viewport);

    setCentralWidget(splitter.get());

    viewport = gl_viewport;
    viewport->set_render_queue_filler([this](gfx::Render_Queue* rq) { render_world(rq); });

    connect(&render_timer, SIGNAL(timeout()), viewport, SLOT(update()));
    render_timer.start(13);

    connect(sim_control->sliderSimSpeed, &QSlider::valueChanged, [&](int value) {
        char buf[64];
        auto speed = value / 4.0f;
        auto res = snprintf(buf, 63, "%.2fx", speed);
        buf[res] = 0;
        sim_control->lblSpeedValue->setText((char const*)buf);
        this->sim_speed = speed;
    });

    connect(sim_control->btnStart, SIGNAL(released()), this, SLOT(start_simulation()));
    connect(sim_control->btnStop, SIGNAL(released()), this, SLOT(stop_simulation()));
    connect(sim_control->btnReset, SIGNAL(released()), this, SLOT(reset_simulation()));

    extensions.insert("#0 None", sb::Extension::None);
    extensions.insert("#1 Rope demo", sb::Extension::Debug_Rope);
    extensions.insert("#2 Cloth demo", sb::Extension::Debug_Cloth);
    extensions.insert("#3 Plant sim", sb::Extension::Plant_Simulation);

    sim_config->cbExtension->addItems(QStringList(extensions.keys()));
    sim_config->cbExtension->setCurrentIndex(0);

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

    connect(sim_config->cbExtension, &QComboBox::currentTextChanged, this, &Window_Main::on_extension_changed);

    // Default sim config
    auto extension = sim_cfg.property("ext");
    sim_cfg.setProperty("ext", "Plant_Simulation");
    extension = sim_cfg.property("ext");
    sim_cfg.seed_position = Vec3(0, 0, 0);
    sim_cfg.setProperty("density", 1.0f);
    sim_cfg.setProperty("attachment_strength", 1.0f);
    sim_cfg.setProperty("surface_adaption_strength", 1.0f);
    sim_cfg.setProperty("stiffness", 0.2f);
    sim_cfg.setProperty("aging_rate", 0.1f);
    sim_cfg.setProperty("phototropism_response_strength", 1.0f);
    sim_cfg.setProperty("branching_probability", 0.25f);
    sim_cfg.setProperty("branch_angle_variance", glm::pi<float>());
    sim_cfg.setProperty("particle_count_limit", 128u);
    
    // Not settable from UI
    sim_cfg.compute_preference = sb::Compute_Preference::GPU_Proprietary;

    connect(sim_control->btnSaveImage, &QPushButton::released, [&]() {
        stop_simulation();
        auto path = QFileDialog::getSaveFileName(this, tr("Save simulation image"), QString(), "Simulation image (*.simg);;All files (*.*)");

        if (path.isEmpty()) {
            return;
        }

        auto ser = make_serializer(path);

        if (!simulation->save_image(ser.get())) {
            QMessageBox::critical(this, "Save image error", "Couldn't save simulation image!");
        }
    });

    connect(sim_control->btnLoadImage, &QPushButton::released, [&]() {
        stop_simulation();

        auto path = QFileDialog::getOpenFileName(this, tr("Load simulation image"), QString(), "Simulation image (*.simg);;All files (*.*)");

        if (path.isEmpty()) {
            return;
        }

        auto ser = make_deserializer(path);

        if (!simulation) {
            simulation = sb::create_simulation(sim_cfg);
        }

        if (!simulation->load_image(ser.get())) {
            QMessageBox::critical(this, "Load image error", "Couldn't load simulation image!");
        }
    });

    connect(sim_control->btnMeshgen, &QPushButton::released, [&]() {
        auto wnd = new Window_Meshgen(simulation);
        connect(this, &Window_Main::render, wnd, &Window_Meshgen::render);
        wnd->show();
    });

    connect(sim_control->btnLoadObj, &QPushButton::released, [&]() {
        try_load_mesh();
    });
}

void Window_Main::start_simulation() {
    if (!is_simulation_running()) {
        if (!simulation) {
            reset_simulation();
        }
        conn_sim_step = connect(&render_timer, SIGNAL(timeout()), this, SLOT(step_simulation()));
    }
}

void Window_Main::render_world(gfx::Render_Queue* rq) {
    assert(rq != NULL);
    render_softbody_simulation(rq, simulation.get(), render_params);
    emit render(rq);
}

void Window_Main::stop_simulation() {
    if (is_simulation_running()) {
        disconnect(*conn_sim_step);
        conn_sim_step.reset();
    }
}

void Window_Main::reset_simulation() {
    simulation = sb::create_simulation(sim_cfg, sb_msg_callback);
    sb::sdf::ast::Expression<float>* expr;
    sb::sdf::ast::Sample_Point* sp;
    collider_builder->get_ast(&expr, &sp);
    sb::ISoftbody_Simulation::Collider_Handle h;
    simulation->add_collider(h, expr, sp);
}

void Window_Main::step_simulation() {
    if (simulation) {
        simulation->step(sim_speed * render_timer.interval() / 1000.0f);
    }
}

void Window_Main::on_extension_changed(QString const& k) {
    sim_cfg.ext = extensions.value(sim_config->cbExtension->currentText(), sb::Extension::None);
}

bool Window_Main::is_simulation_running() {
    return simulation != nullptr && conn_sim_step.has_value();
}

void Window_Main::try_load_mesh() {
    if (simulation == nullptr) {
        QMessageBox(QMessageBox::Icon::Critical, "Error", "No simulation!").exec();
        return;
    }

    if (conn_sim_step.has_value()) {
        QMessageBox(QMessageBox::Icon::Critical, "Error", "Simulation is already in progress!").exec();
        return;
    }

    auto path = QFileDialog::getOpenFileName(this, tr("Load an *.obj file into the simulator"), QString(), "Wavefront mesh (*.obj);;All files (*.*)");

    if (path.isEmpty()) {
        return;
    }

    auto path_arr = path.toLocal8Bit();

    objscan_extra ex;
    // TODO(danielm): users should be able to set this
    ex.subdivisions = 16;

    objscan_result res;
    res.extra = &ex;
    if (!objscan_from_obj_file(&res, path_arr.data())) {
        QMessageBox(QMessageBox::Icon::Critical, "Error", "Failed to load Wavefront mesh '%s'!").exec();
        return;
    }

    if (res.particle_count > 0) {
        simulation->add_particles(res.particle_count, (glm::vec4*)res.positions);
        simulation->add_connections(res.connection_count, (long long*)res.connections);
    }

    objscan_free_result(&res);
}
