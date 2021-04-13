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

	auto pagePlant = new Wizard_Page_Plant(this);
	connectSpinBox(pagePlant->ui()->sbOriginX, &_sbExtPlant.seed_position.x);
	connectSpinBox(pagePlant->ui()->sbOriginY, &_sbExtPlant.seed_position.y);
	connectSpinBox(pagePlant->ui()->sbOriginZ, &_sbExtPlant.seed_position.z);
	connectSpinBox(pagePlant->ui()->sbParticleCountLimit, &_sbExtPlant.particle_count_limit);
	connectSpinBox(pagePlant->ui()->sbStiffness, &_sbExtPlant.stiffness);
	connectSpinBox(pagePlant->ui()->sbDensity, &_sbExtPlant.density);
	connectSpinBox(pagePlant->ui()->sbPhotoRespStr, &_sbExtPlant.phototropism_response_strength);
	connectSpinBox(pagePlant->ui()->sbAgingRate, &_sbExtPlant.aging_rate);
	connectSpinBox(pagePlant->ui()->sbBranchProb, &_sbExtPlant.branching_probability);
	connectSpinBox(pagePlant->ui()->sbBranchVar, &_sbExtPlant.branch_angle_variance);
	connectSpinBox(pagePlant->ui()->sbSurfAdaptStr, &_sbExtPlant.surface_adaption_strength);
	connectSpinBox(pagePlant->ui()->sbSurfAttachStr, &_sbExtPlant.attachment_strength);
	setPage(Page_Config_Plant, pagePlant);

	// setPage(Page_Config_Cloth, new Wizard_Page_Cloth(this));
	
	auto pageCompute = new Wizard_Page_Compute(this);
	connectRadioButtonToEnumField(pageCompute->ui()->btnNone, &_sbConfig.compute_preference, sb::Compute_Preference::None);
	connectRadioButtonToEnumField(pageCompute->ui()->btnCUDA, &_sbConfig.compute_preference, sb::Compute_Preference::GPU_Proprietary);
	connectRadioButtonToEnumField(pageCompute->ui()->btnOpenCL, &_sbConfig.compute_preference, sb::Compute_Preference::GPU_OpenCL);
	connectRadioButtonToEnumField(pageCompute->ui()->btnCPU, &_sbConfig.compute_preference, sb::Compute_Preference::Reference);
	setPage(Page_Compute, pageCompute);

	setPage(Page_Outro, new Wizard_Page_Outro(this));

	setWindowTitle("Create a softbody simulation");
	setStartId(Page_Intro);
}

void Wizard_SB_Simulation::accept() {
	if (field(fieldSrcFromScratch).toBool()) {
		if (field(fieldExtensionPlant).toBool()) {
			_sbConfig.ext = sb::Extension::Plant_Simulation;
			_sbConfig.extra.plant_sim = &_sbExtPlant;
		} else if (field(fieldExtensionCloth).toBool()) {
			_sbConfig.ext = sb::Extension::Debug_Cloth;
			_sbConfig.extra.cloth_sim = &_sbExtCloth;
		}

		QDialog::accept();
	} else {
		assert(0);
	}
}
