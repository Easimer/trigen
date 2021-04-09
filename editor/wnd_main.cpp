// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#include "stdafx.h"
#include "wnd_main.h"
#include "ui_wnd_main.h"
#include "wizard_sb_simulation.h"
#include <QMessageBox>
#include <QListView>

Window_Main::Window_Main(QWidget *parent) :
    QMainWindow(parent),
    _ui(new Ui::Window_Main()),
    _splitter(Qt::Orientation::Horizontal, this) {
    _ui->setupUi(this);

    setCentralWidget(&_splitter);
    _splitter.setChildrenCollapsible(false);
    // Placeholder items in the splitter
    _splitter.insertWidget(0, new QWidget(this));
    _splitter.insertWidget(1, new QWidget(this));

    connect(_ui->actionNew, &QAction::triggered, this, [this]() { newSession("TEST"); });

    connect(&_vm, &VM_Main::currentSessionChanged, this, &Window_Main::currentSessionChanged);

    _ui->toolBar->addAction(QIcon(":/images/add_plant.svg"), "Add softbody", [this]() {
        auto wizard = new Wizard_SB_Simulation(this);

        wizard->show();
        connect(wizard, &Wizard_SB_Simulation::accepted, [&, wizard]() {
            auto &cfg = wizard->config();
            _vm.addSoftbodySimulation(cfg);
        });
    });

    // Disable all toolbar buttons
    for (auto &action : _ui->toolBar->actions()) {
        action->setEnabled(false);
    }
}

Window_Main::~Window_Main() {
    delete _ui;
}

void Window_Main::setViewport(Filament_Viewport *viewport) {
    _splitter.insertWidget(0, viewport);
    viewport->setSizePolicy(QSizePolicy(QSizePolicy::Policy::Maximum, QSizePolicy::Policy::Maximum));
    _viewport = viewport;

    connect(_viewport, &Filament_Viewport::onMouseDown, &_vm, &VM_Main::onMouseDown);
    connect(_viewport, &Filament_Viewport::onMouseUp, &_vm, &VM_Main::onMouseUp);
    connect(_viewport, &Filament_Viewport::onMouseMove, &_vm, &VM_Main::onMouseMove);
    connect(_viewport, &Filament_Viewport::onMouseWheel, &_vm, &VM_Main::onMouseWheel);
    connect(_viewport, &Filament_Viewport::onWindowResize, &_vm, &VM_Main::onWindowResize);
    connect(&_vm, &VM_Main::cameraUpdated, _viewport, &Filament_Viewport::updateCamera);
}

void Window_Main::newSession(QString const &name) {
    auto nameUtf8 = name.toUtf8();
    auto session = _vm.createNewSession(nameUtf8.constData());

    _ui->menuWindow->addAction(name, [this, session]() {
        _vm.switchToSession(session);
    });
}

void Window_Main::currentSessionChanged(Session *session) {
    if (session != nullptr) {
        for (auto &action : _ui->toolBar->actions()) {
            action->setEnabled(true);
        }
    }
}