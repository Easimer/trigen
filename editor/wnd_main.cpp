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

    _ui->toolBar->addAction(QIcon(":/images/add_plant.svg"), "Add softbody", [this]() {
        auto wizard = new Wizard_SB_Simulation(this);

        wizard->show();
        connect(wizard, &Wizard_SB_Simulation::accepted, [&]() {
            auto &cfg = wizard->config();
            if (_currentSession != nullptr) {
                _currentSession->createPlant(cfg);
            } else {
                QMessageBox::critical(wizard, "Couldn't create softbody", "Couldn't create softbody: no active session!");
            }
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
}

void Window_Main::newSession(QString const &name) {
    auto nameUtf8 = name.toUtf8();
    auto session = std::make_unique<Session>(nameUtf8.constData());
    auto sessionPtr = session.get();

    _ui->menuWindow->addAction(name, [this, sessionPtr]() {
        switchToSession(sessionPtr);
    });

    connect(_viewport, &Filament_Viewport::onMouseDown, sessionPtr, &Session::onMouseDown);
    connect(_viewport, &Filament_Viewport::onMouseUp, sessionPtr, &Session::onMouseUp);
    connect(_viewport, &Filament_Viewport::onMouseMove, sessionPtr, &Session::onMouseMove);
    connect(_viewport, &Filament_Viewport::onMouseWheel, sessionPtr, &Session::onMouseWheel);
    connect(_viewport, &Filament_Viewport::onWindowResize, sessionPtr, &Session::onWindowResize);
    connect(sessionPtr, &Session::viewMatrixUpdated, _viewport, &Filament_Viewport::updateViewMatrix);

    _currentSession = sessionPtr;
    _sessions.emplace_back(std::move(session));
}

void Window_Main::switchToSession(Session *session) {
    _currentSession = session;
    
    if (session != nullptr) {
        for (auto &action : _ui->toolBar->actions()) {
            action->setEnabled(true);
        }
    }
}