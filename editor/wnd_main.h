// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: main window declaration
//

#pragma once

#include <list>
#include <memory>
#include <QMainWindow>
#include <QLayout>
#include <QSplitter>
#include "session.h"
#include "ui_wnd_main.h"
#include "filament_viewport.h"

namespace Ui {
class Window_Main;
}

class Window_Main : public QMainWindow {
    Q_OBJECT;
public:
    explicit Window_Main(QWidget *parent = nullptr);
    ~Window_Main();

    void setViewport(Filament_Viewport *viewport);

    void newSession(QString const &name);

public slots:
    void switchToSession(Session *ptr);

private:
    Ui::Window_Main *_ui;
    std::list<std::unique_ptr<Session>> _sessions;
    Session *_currentSession = nullptr;
    Filament_Viewport *_viewport = nullptr;
    QSplitter _splitter;
};
