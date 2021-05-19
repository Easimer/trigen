// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: Configuration wizard used when creating a softbody
//          simulation
//

#pragma once

#include <QWizard>
#include <QDoubleSpinBox>
#include <QSpinBox>
#include <QRadioButton>
#include <trigen.hpp>

class Wizard_SB_Simulation : public QWizard {
	Q_OBJECT;
public:
	Wizard_SB_Simulation(QWidget *parent = nullptr, Qt::WindowFlags flags = 0);

	Trigen_Parameters const &config() {
		return _sbConfig;
	}

	void accept() override;

private:
	template<typename T>
	QMetaObject::Connection connectSpinBox(QDoubleSpinBox *sb, T *v);
	template<typename T>
	QMetaObject::Connection connectSpinBox(QSpinBox *sb, T *v);
	template<typename T>
	QMetaObject::Connection connectRadioButtonToEnumField(QRadioButton *btn, T *field, T value);

private:
	Trigen_Parameters _sbConfig = {};
	sb::Plant_Simulation_Extension_Extra _sbExtPlant = {};
	sb::Debug_Cloth_Extension_Extra _sbExtCloth = {};
};
