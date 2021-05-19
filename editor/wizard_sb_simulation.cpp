// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#include "stdafx.h"
#include "wizard_sb_simulation.h"
#include "wizard_sb_simulation_pages.h"

template<typename T>
inline QMetaObject::Connection Wizard_SB_Simulation::connectSpinBox(QDoubleSpinBox *sb, T *v) {
	*v = sb->value();

	return connect(sb, static_cast<void (QDoubleSpinBox:: *)(double)>(&QDoubleSpinBox::valueChanged), [v](double nv) {
		*v = nv;
	});
}

template<typename T>
inline QMetaObject::Connection Wizard_SB_Simulation::connectSpinBox(QSpinBox *sb, T *v) {
	*v = sb->value();

	return connect(sb, static_cast<void (QSpinBox:: *)(int)>(&QSpinBox::valueChanged), [v](int nv) {
		*v = nv;
	});
}

template<typename T>
inline QMetaObject::Connection Wizard_SB_Simulation::connectRadioButtonToEnumField(QRadioButton *btn, T *field, T value) {
	if (btn->isChecked()) {
		*field = value;
	}

	return connect(btn, &QRadioButton::clicked, [field, value](bool checked) {
		if (checked) {
			*field = value;
		}
	});
}

Wizard_SB_Simulation::Wizard_SB_Simulation(QWidget *parent, Qt::WindowFlags flags) : QWizard(parent, flags) {
	setPage(Page_Intro, new Wizard_Page_Intro(this));
	setPage(Page_Type, new Wizard_Page_Type(this));

    _sbConfig.flags = 0;

	auto pagePlant = new Wizard_Page_Plant(this);
	connectSpinBox(pagePlant->ui()->sbOriginX, &_sbConfig.seed_position[0]);
	connectSpinBox(pagePlant->ui()->sbOriginY, &_sbConfig.seed_position[1]);
	connectSpinBox(pagePlant->ui()->sbOriginZ, &_sbConfig.seed_position[2]);
	connectSpinBox(pagePlant->ui()->sbParticleCountLimit, &_sbConfig.particle_count_limit);
	connectSpinBox(pagePlant->ui()->sbStiffness, &_sbConfig.stiffness);
	connectSpinBox(pagePlant->ui()->sbDensity, &_sbConfig.density);
	connectSpinBox(pagePlant->ui()->sbPhotoRespStr, &_sbConfig.phototropism_response_strength);
	connectSpinBox(pagePlant->ui()->sbAgingRate, &_sbConfig.aging_rate);
	connectSpinBox(pagePlant->ui()->sbBranchProb, &_sbConfig.branching_probability);
	connectSpinBox(pagePlant->ui()->sbBranchVar, &_sbConfig.branch_angle_variance);
	connectSpinBox(pagePlant->ui()->sbSurfAdaptStr, &_sbConfig.surface_adaption_strength);
	connectSpinBox(pagePlant->ui()->sbSurfAttachStr, &_sbConfig.attachment_strength);
	setPage(Page_Config_Plant, pagePlant);

	// setPage(Page_Config_Cloth, new Wizard_Page_Cloth(this));
	
	auto pageCompute = new Wizard_Page_Compute(this);
    _sbConfig.flags |= (pageCompute->ui()->chkPreferCPU->isChecked() ? Trigen_F_PreferCPU : 0);
    connect(pageCompute->ui()->chkPreferCPU, &QCheckBox::stateChanged, [&](int state) {
		if (state == 0) {
            _sbConfig.flags |= Trigen_F_PreferCPU;
		} else {
            _sbConfig.flags &= ~(Trigen_F_PreferCPU);
		}
    });
	setPage(Page_Compute, pageCompute);

	setPage(Page_Outro, new Wizard_Page_Outro(this));

	setWindowTitle("Create a softbody simulation");
	setStartId(Page_Intro);
}

void Wizard_SB_Simulation::accept() {
	if (field(fieldSrcFromScratch).toBool()) {
		QDialog::accept();
	} else {
		assert(0);
	}
}
