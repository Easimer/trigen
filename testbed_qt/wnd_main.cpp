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

#define BIND_DATA_DBLSPINBOX(data, Data_Type, field, elem)                                                  \
    connect(                                                                                                \
        &data, &Data_Type::field##_changed,                                                                 \
        sim_config->elem, &QDoubleSpinBox::setValue                                                         \
    );                                                                                                      \
    connect(                                                                                                \
        sim_config->elem, static_cast<void (QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged),     \
        &data, [&](float v) { sim_cfg.setProperty(#field, v); }                                             \
    );                                                                                                      \

#define BIND_DATA_DBLSPINBOX_PLANTSIM(field, elem) \
    BIND_DATA_DBLSPINBOX(plant_sim_cfg, Plant_Simulation_Extension_Extra, field, elem)

#define BIND_DATA_SPINBOX(data, Data_Type, field, elem)                                     \
    connect(                                                                                \
        &data, &Data_Type::field##_changed,                                                 \
        sim_config->elem, &QSpinBox::setValue                                               \
    );                                                                                      \
    connect(                                                                                \
        sim_config->elem, static_cast<void (QSpinBox::*)(int )>(&QSpinBox::valueChanged),   \
        &data, [&](int v) { data.setProperty(#field, (unsigned)v); }                        \
    );                                                                                      \

#define BIND_DATA_SPINBOX_PLANTSIM(field, elem) \
    BIND_DATA_SPINBOX(plant_sim_cfg, Plant_Simulation_Extension_Extra, field, elem)
#define BIND_DATA_SPINBOX_CLOTHSIM(field, elem) \
    BIND_DATA_SPINBOX(cloth_sim_cfg, Debug_Cloth_Extension_Extra, field, elem)

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
        &obj_field, &QVec3::sig,                                                                         \
        sim_config->field, &QDoubleSpinBox::setValue);                                                   \
    connect(                                                                                             \
        sim_config->field, static_cast<void (QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged), \
        &obj_field, &QVec3::set);                                                                        \

#define BIND_DATA_VEC3(vec, fieldx, fieldy, fieldz)             \
    BIND_DATA_VEC3_COMPONENT(vec, fieldx, x_changed, set_x);    \
    BIND_DATA_VEC3_COMPONENT(vec, fieldy, y_changed, set_y);    \
    BIND_DATA_VEC3_COMPONENT(vec, fieldz, z_changed, set_z);    \

std::unique_ptr<sb::ISerializer> make_serializer(QString const& path);
std::unique_ptr<sb::IDeserializer> make_deserializer(QString const& path);

static void sb_msg_callback(sb::Debug_Message_Source src, sb::Debug_Message_Type type, sb::Debug_Message_Severity severity, char const* msg, void* user) {
    printf("sb: %s\n", msg);

    if (user != nullptr) {
        reinterpret_cast<Window_Main*>(user)->on_debug_output(src, type, severity, msg);
    }
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
    splitter->setChildrenCollapsible(false);
    splitter->addWidget(tabs);
    splitter->addWidget(gl_viewport);

    setCentralWidget(splitter.get());

    viewport = gl_viewport;
    dbg_visualizer = make_debug_visualizer();
    viewport->set_render_queue_filler([this](gfx::Render_Queue* rq) { render_world(rq); });

    // Resize the window so that the GL viewport is visible by default
    setMinimumWidth(width() * 1.25f);

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

    BIND_DATA_SPINBOX_PLANTSIM(particle_count_limit, sbParticleCountLimit);
    BIND_DATA_DBLSPINBOX_PLANTSIM(density, sbDensity);
    BIND_DATA_DBLSPINBOX_PLANTSIM(attachment_strength, sbAttachmentStrength);
    BIND_DATA_DBLSPINBOX_PLANTSIM(surface_adaption_strength, sbAdaptionStrength);
    BIND_DATA_DBLSPINBOX_PLANTSIM(stiffness, sbStiffness);
    BIND_DATA_DBLSPINBOX_PLANTSIM(aging_rate, sbAgingRate);
    BIND_DATA_DBLSPINBOX_PLANTSIM(phototropism_response_strength, sbPhototropismResponseStrength);
    BIND_DATA_DBLSPINBOX_PLANTSIM(branching_probability, sbBranchingProbability);
    BIND_DATA_DBLSPINBOX_PLANTSIM(branch_angle_variance, sbBranchAngleVariance);
    BIND_DATA_VEC3(plant_sim_cfg.seed_position, sbOriginX, sbOriginY, sbOriginZ);

    BIND_DATA_SPINBOX_CLOTHSIM(dim, sbClothSimDim);

    connect(sim_config->cbExtension, &QComboBox::currentTextChanged, this, &Window_Main::on_extension_changed);

    // Default sim config
    auto extension = sim_cfg.property("ext");
    sim_cfg.setProperty("ext", "Plant_Simulation");
    extension = sim_cfg.property("ext");

    cloth_sim_cfg.setProperty("dim", 64);

    plant_sim_cfg.seed_position = Vec3(0, 0, 0);
    plant_sim_cfg.setProperty("density", 1.0f);
    plant_sim_cfg.setProperty("attachment_strength", 1.0f);
    plant_sim_cfg.setProperty("surface_adaption_strength", 1.0f);
    plant_sim_cfg.setProperty("stiffness", 0.2f);
    plant_sim_cfg.setProperty("aging_rate", 0.1f);
    plant_sim_cfg.setProperty("phototropism_response_strength", 1.0f);
    plant_sim_cfg.setProperty("branching_probability", 0.25f);
    plant_sim_cfg.setProperty("branch_angle_variance", glm::pi<float>());
    plant_sim_cfg.setProperty("particle_count_limit", 128u);

    sim_cfg.extra.ptr = nullptr;
    
    // Not settable from UI
    // sim_cfg.compute_preference = sb::Compute_Preference::GPU_Proprietary;
    sim_cfg.compute_preference = sb::Compute_Preference::Reference;

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
        if (simulation != nullptr) {
            auto wnd = make_meshgen_window(simulation, nullptr);
            // FUCK
            connect(this, SIGNAL(render(gfx::Render_Queue*)), wnd, SLOT(render(gfx::Render_Queue*)));
            wnd->show();
        }
    });

    connect(sim_control->btnLoadObj, &QPushButton::released, [&]() {
        try_load_mesh_into_simulation();
    });

    connect(sim_control->btnLoadObjColl, &QPushButton::released, [&]() {
        try_load_mesh_collider();
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

    if (model != nullptr) {
        // Iterate over mesh colliders and render them
        for (int collider_idx = 0; collider_idx < model->num_mesh_colliders(); collider_idx++) {
            sb::Mesh_Collider coll;
            if (model->mesh_collider(&coll, collider_idx)) {
                render_mesh_collider(rq, &coll);
            }
        }
    }

    if (sim_control->chkDrawDebugVis->isChecked()) {
        gfx::allocate_command_and_initialize<Subqueue_Render_Command>(rq, dbg_visualizer.get());
    }

    emit render(rq);
}

void Window_Main::stop_simulation() {
    if (is_simulation_running()) {
        disconnect(*conn_sim_step);
        conn_sim_step.reset();
    }
}

void Window_Main::reset_simulation() {
    simulation = sb::create_simulation(sim_cfg, sb_msg_callback, this);
    model = make_viewmodel_main(simulation.get());
    sb::sdf::ast::Expression<float>* expr;
    sb::sdf::ast::Sample_Point* sp;
    collider_builder->get_ast(&expr, &sp);
    sb::ISoftbody_Simulation::Collider_Handle h;
    simulation->add_collider(h, expr, sp);
    simulation->set_debug_visualizer(dbg_visualizer.get());
}

void Window_Main::step_simulation() {
    if (simulation) {
        simulation->step(sim_speed * render_timer.interval() / 1000.0f);
    }
}

void Window_Main::on_extension_changed(QString const& k) {
    sim_cfg.ext = extensions.value(sim_config->cbExtension->currentText(), sb::Extension::None);

    // Reset extra pointer
    switch(sim_cfg.ext) {
        case sb::Extension::Plant_Simulation:
            sim_cfg.extra.plant_sim = &plant_sim_cfg;
            break;
        case sb::Extension::Debug_Cloth:
            sim_cfg.extra.cloth_sim = &cloth_sim_cfg;
            break;
        default:
            sim_cfg.extra.ptr = nullptr;
            break;
    }
}

bool Window_Main::is_simulation_running() {
    return simulation != nullptr && conn_sim_step.has_value();
}

void Window_Main::on_debug_output(sb::Debug_Message_Source src, sb::Debug_Message_Type type, sb::Debug_Message_Severity sever, char const* msg) {
    if (type == sb::Debug_Message_Type::Benchmark) {
        // TODO(danielm): route benchmark messages to a separate benchmark info window
        return;
    }

    char* buf = NULL;
    int n;
    size_t siz = 0;

    n = snprintf(buf, siz, "sb: %s\n", msg);

    if (n < 0) {
        return;
    }

    siz = (size_t)n + 1;
    buf = new char[siz];
    if (buf == nullptr) {
        return;
    }

    n = snprintf(buf, siz, "sb: %s\n", msg);

    if (n < 0) {
        delete[] buf;
        return;
    }

    QString const& currentOutput = sim_control->simOutput->toPlainText();
    int currentLength = currentOutput.length();
    int newLength = currentLength + n;
    auto newOutput = currentOutput + (char const*)buf;
    sim_control->simOutput->setPlainText(newOutput);
}

bool Window_Main::ask_user_for_path_to_mesh(QByteArray &out_path) {
    auto path = QFileDialog::getOpenFileName(this, tr("Load an *.obj file into the simulator"), QString(), "Wavefront mesh (*.obj);;All files (*.*)");

    if (path.isEmpty()) {
        return false;
    }

    out_path = path.toLocal8Bit();
    return true;
}

void Window_Main::try_load_mesh_into_simulation() {
    if (simulation == nullptr) {
        QMessageBox(QMessageBox::Icon::Critical, "Error", "No simulation!").exec();
        return;
    }

    if (conn_sim_step.has_value()) {
        QMessageBox(QMessageBox::Icon::Critical, "Error", "Simulation is already in progress!").exec();
        return;
    }

    QByteArray path_arr;
    if (!ask_user_for_path_to_mesh(path_arr)) {
        return;
    }

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

void Window_Main::try_load_mesh_collider() {
    if (simulation == nullptr) {
        QMessageBox(QMessageBox::Icon::Critical, "Error", "No simulation!").exec();
        return;
    }

    if (conn_sim_step.has_value()) {
        QMessageBox(QMessageBox::Icon::Critical, "Error", "Simulation is already in progress!").exec();
        return;
    }

    QByteArray path_arr;
    if (!ask_user_for_path_to_mesh(path_arr)) {
        return;
    }

    std::string err_msg;
    if (!model->add_mesh_collider(path_arr, err_msg)) {
        QMessageBox(QMessageBox::Icon::Critical, "Error loading mesh", err_msg.c_str()).exec();
    }
}
