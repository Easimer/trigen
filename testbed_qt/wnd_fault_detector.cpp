// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: UI for Fault_Detector
//

#include "wnd_fault_detector.h"

Window_Fault_Detector::Window_Fault_Detector(QMainWindow* parent) : QDialog(parent), timer(this) {
    layout.addWidget((QWidget*)ui_ex);

    btn_step.setText("Step");
    connect(&btn_step, &QPushButton::pressed, this, &Window_Fault_Detector::step_simulations);
    layout.addWidget(&btn_step);

    chk_auto.setText("Auto-tick (NYI)");
    connect(&chk_auto, &QCheckBox::toggled, this, &Window_Fault_Detector::toggle_timer);
    layout.addWidget(&chk_auto);

    setLayout(&layout);
}

static QString format_exception(Fault_Detector_Exception const& ex) {
    char buffer[512];

    auto idx = snprintf(buffer, 511, "%s\n\nBackend: %d\nAt step: '%s'\n", ex.what(), ex.backend(), ex.step());
    buffer[idx] = 0;

    return QString((char const*)buffer);
}

void Window_Fault_Detector::step_simulations() {
    try {
        fd.step();
    } catch (Fault_Detector_Exception const& ex) {
        ui_ex->editException->setPlainText(format_exception(ex));
        toggle_timer(false);
    }
}

void Window_Fault_Detector::toggle_timer(bool b) {
    printf("fd: auto-ticking enabled=%d\n", (int)b);

    if (!b && timer_conn.has_value()) {
        disconnect(*timer_conn);
        timer_conn.reset();
        timer.stop();
    } else if(b && !timer_conn.has_value()) {
        timer_conn = connect(&timer, &QTimer::timeout, this, &Window_Fault_Detector::step_simulations);
        timer.start(33);
    }
}
