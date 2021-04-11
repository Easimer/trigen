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
#include "vm_main.h"

namespace Ui {
class Window_Main;
}

class Window_Main : public QMainWindow {
    Q_OBJECT;
public:
    explicit Window_Main(std::unique_ptr<VM_Main> &&vm, QWidget *parent = nullptr);
    ~Window_Main();

    void setViewport(Filament_Viewport *viewport);

    void newSession(QString const &name);

public slots:
    void currentSessionChanged(Session *ptr);

private:
    Ui::Window_Main *_ui;
    std::unique_ptr<VM_Main> _vm;
    Filament_Viewport *_viewport = nullptr;
    QSplitter _splitter;
};