// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: entry point
//

#include "common.h"
#include "mainwindow.h"
#include <QApplication>

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    MainWindow wnd;
    wnd.setAnimated(true);
    wnd.show();
    return app.exec();
}