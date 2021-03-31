// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: main window declaration
//

#pragma once

#include <vector>
#include <memory>
#include <QMainWindow>
#include <QLayout>
#include "session.h"
#include "ui_wnd_main.h"

namespace Ui {
class Window_Main;
}

class Window_Main : public QMainWindow {
    Q_OBJECT;
public:
    explicit Window_Main(QWidget *parent = nullptr);
    ~Window_Main();

    void setViewport(QWidget *viewport);

private:
    Ui::Window_Main *_ui;
    std::vector<std::unique_ptr<Session>> _sessions;
};
