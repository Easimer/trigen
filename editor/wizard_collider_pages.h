// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: Wizard pages used by Wizard_Collider
//

#pragma once

#include <QWizardPage>
#include <QFileDialog>
#include <QAction>
#include "ui_wizard_collider_browse.h"

class Wizard_Page_Browse : public QWizardPage {
	Q_OBJECT;
public:
	Wizard_Page_Browse(QWidget *parent = nullptr) : QWizardPage(parent) {
		_ui.setupUi(this);

		connect(_ui.actionBrowse, &QAction::triggered, [&]() {
            auto path = QFileDialog::getOpenFileName(this, tr("Choose a collider mesh..."), QString(), "Wavefront mesh (*.obj);;All files (*.*)");

            if (path.isEmpty()) {
                return false;
            }

			_ui.editPath->setText(path);
        });
		registerField("fieldPath", _ui.editPath);
	}

	QString path() const {
		return _ui.editPath->text();
	}
private:
	Ui::Wizard_Collider_Browse _ui;
};

