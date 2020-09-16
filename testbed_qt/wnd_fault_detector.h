// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: UI for Fault_Detector
//

#pragma once

#include "common.h"
#include <memory>
#include <QWindow>
#include <QLayout>
#include <QCheckBox>
#include <QTimer>
#include <QPushButton>
#include "fault_detector.h"
#include "ui_sb_fault_detector_exception.h"
#include "helper_ui_widget.h"

class Window_Fault_Detector : public QDialog {
    Q_OBJECT;
public:
    Window_Fault_Detector(
        QMainWindow* parent = nullptr
    );

public slots:
    void step_simulations();
    void toggle_timer(bool b);

private:
    QHBoxLayout layout;
    QPushButton btn_step;
    QTimer timer;
    Optional<QMetaObject::Connection> timer_conn;
    QCheckBox chk_auto;
    Ui_Widget<Ui::Fault_Detector_Exception> ui_ex;

    Fault_Detector fd;
};
