// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#include "stdafx.h"
#include "wnd_main.h"
#include "ui_wnd_main.h"
#include "wizard_sb_simulation.h"
#include "wizard_collider.h"
#include <QMessageBox>
#include <QListView>
#include "entity_list.h"

Window_Main::Window_Main(std::unique_ptr<VM_Main> &&vm, std::unique_ptr<QAbstractItemModel> &&entityListModel, QWidget *parent) :
    QMainWindow(parent),
    _ui(new Ui::Window_Main()),
    _vm(std::move(vm)),
    _splitter(Qt::Orientation::Horizontal, this),
    _viewport(this),
    _entityList(this),
    _entityListModel(std::move(entityListModel)) {
    _ui->setupUi(this);

    setCentralWidget(&_splitter);
    _splitter.setChildrenCollapsible(false);
    _splitter.insertWidget(0, &_viewport);
    _splitter.insertWidget(1, &_entityList);

    _entityList.setModel(_entityListModel.get());

    connect(_ui->actionNew, &QAction::triggered, this, [this]() { newSession("TEST"); });

    connect(_vm.get(), &VM_Main::currentSessionChanged, this, &Window_Main::currentSessionChanged);

    connect(_ui->actionCreateSoftbody, &QAction::triggered, [&]() {
        auto wizard = new Wizard_SB_Simulation(this);

        wizard->show();
        connect(wizard, &Wizard_SB_Simulation::accepted, [&, wizard]() {
            auto &cfg = wizard->config();
            _vm->addSoftbodySimulation(cfg);
        });
    });

    connect(_ui->actionCreateCollider, &QAction::triggered, [&]() {
        auto wizard = new Wizard_Collider(this);

        wizard->show();
        connect(wizard, &Wizard_Collider::accepted, [&, wizard]() {
            auto path = wizard->path().toUtf8();
            _vm->addColliderFromPath(path.constData());
        });
    });

    connect(_ui->actionStart, &QAction::triggered, [&]() {
        _ui->actionPause->setEnabled(true);
        _ui->actionStart->setEnabled(false);
        _vm->setRunning(true);
    });

    connect(_ui->actionPause, &QAction::triggered, [&]() {
        _ui->actionPause->setEnabled(false);
        _ui->actionStart->setEnabled(true);
        _vm->setRunning(false);
    });

    connect(_ui->actionModeTranslate, &QAction::triggered, [&]() {
        _vm->setGizmoMode(Session_Gizmo_Mode::Translation);
    });

    connect(_ui->actionModeRotate, &QAction::triggered, [&]() {
        _vm->setGizmoMode(Session_Gizmo_Mode::Rotation);
    });

    connect(_ui->actionModeScale, &QAction::triggered, [&]() {
        _vm->setGizmoMode(Session_Gizmo_Mode::Scaling);
    });

    // Enable/disable the start/pause button when the user changes the current session
    connect(_vm.get(), &VM_Main::currentSessionChanged, [&](Session *session) {
        _ui->actionStart->setEnabled(!session->isRunning());
        _ui->actionPause->setEnabled(session->isRunning());
    });

    // Disable all toolbar buttons
    for (auto &action : _ui->toolBarSide->actions()) {
        action->setEnabled(false);
    }

    connect(&_viewport, &GLViewport::rendering, _vm.get(), &VM_Main::onRender);

    connect(&_renderTimer, SIGNAL(timeout()), &_viewport, SLOT(update()));
    connect(&_renderTimer, &QTimer::timeout, [&]() {
        _vm->onTick(_renderTimer.interval() / 1000.0f);
    });
    _renderTimer.start(13);
}

Window_Main::~Window_Main() {
    delete _ui;
}

void Window_Main::newSession(QString const &name) {
    auto nameUtf8 = name.toUtf8();
    auto session = _vm->createNewSession(nameUtf8.constData());

    _ui->menuWindow->addAction(name, [this, session]() {
        _vm->switchToSession(session);
    });
}

void Window_Main::currentSessionChanged(Session *session) {
    if (session != nullptr) {
        for (auto &action : _ui->toolBarSide->actions()) {
            action->setEnabled(true);
        }
    }
}
