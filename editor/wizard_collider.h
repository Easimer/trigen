// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#pragma once

#include <QWizard>

class Wizard_Collider : public QWizard {
    Q_OBJECT;
public:
	Wizard_Collider(QWidget *parent = nullptr, Qt::WindowFlags flags = 0);
    void accept() override;

    QString const &path() const {
        return _path;
    }

private:
    QString _path;
};
