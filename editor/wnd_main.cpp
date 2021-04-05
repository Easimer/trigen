// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#include "stdafx.h"
#include "wnd_main.h"
#include "ui_wnd_main.h"
#include "wizard_sb_simulation.h"

Window_Main::Window_Main(QWidget *parent) :
    QMainWindow(parent),
    _ui(new Ui::Window_Main()) {
    _ui->setupUi(this);

    connect(_ui->actionNew, &QAction::triggered, this, [this]() { newSession("TEST"); });

    _ui->toolBar->addAction("Add softbody", [this]() {
        auto wizard = new Wizard_SB_Simulation(this);

        wizard->show();
    });
}

Window_Main::~Window_Main() {
    delete _ui;
}

void Window_Main::setViewport(Filament_Viewport *viewport) {
    setCentralWidget(viewport);
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
}