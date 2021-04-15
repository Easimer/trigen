// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#include "stdafx.h"
#include "wizard_collider.h"

#include <QObject>
#include "wizard_collider_pages.h"

Wizard_Collider::Wizard_Collider(QWidget *parent, Qt::WindowFlags flags) : QWizard(parent, flags) {
	setPage(0, new Wizard_Page_Browse(this));

	setWindowTitle("Create a collider");
	setStartId(0);
}

void Wizard_Collider::accept() {
	_path = field("fieldPath").toString();
	QDialog::accept();
}
