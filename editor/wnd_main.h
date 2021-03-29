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

class Window_Main : public QMainWindow {
    Q_OBJECT
public:
    explicit Window_Main(QWidget *parent = nullptr) : QMainWindow(parent) {
        setMinimumSize(QSize(1280, 720));
    }

    void setViewport(QWidget *viewportWidget) {
        setCentralWidget(viewportWidget);
    }

protected:

private:
    std::vector<std::unique_ptr<Session>> _sessions;
};