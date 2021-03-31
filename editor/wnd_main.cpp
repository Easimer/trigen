// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#include "stdafx.h"
#include "wnd_main.h"
#include "ui_wnd_main.h"

Window_Main::Window_Main(QWidget *parent) :
    QMainWindow(parent),
    _ui(new Ui::Window_Main()) {
    _ui->setupUi(this);
}

Window_Main::~Window_Main() {
    delete _ui;
}

void Window_Main::setViewport(QWidget *viewport) {
    setCentralWidget(viewport);
}
