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
#include "ui_wnd_main.h"
#include "vm_main.h"
#include <glviewport.h>
#include <QTimer>

namespace Ui {
class Window_Main;
}

class Window_Main : public QMainWindow {
    Q_OBJECT;
public:
    explicit Window_Main(std::unique_ptr<VM_Main> &&vm, QWidget *parent = nullptr);
    ~Window_Main();

    void newSession(QString const &name);

public slots:
    void currentSessionChanged(Session *session);

private:
    Ui::Window_Main *_ui;
    std::unique_ptr<VM_Main> _vm;
    QSplitter _splitter;
    GLViewport _viewport;
    QTimer _renderTimer;
};
