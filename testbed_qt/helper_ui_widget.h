// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: helper for instantiating UI elements generated from .ui files
//

#pragma once

#include "common.h"
#include <QWidget>

template<typename UI>
class Ui_Widget {
public:
    Ui_Widget(QWidget* parent = nullptr)
        : ui(), widget(std::make_unique<QWidget>(parent)) {
        ui.setupUi(widget.get());
    }

    explicit operator QWidget* () {
        return widget.get();
    }

    UI* operator->() {
        return &ui;
    }
private:
    UI ui;
    Unique_Ptr<QWidget> widget;
};