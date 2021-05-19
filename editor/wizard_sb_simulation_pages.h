// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: Wizard pages used by Wizard_SB_Simulation
//

#pragma once

#include <softbody.h>
#include <QRadioButton>
#include <QButtonGroup>
#include <QLayout>
#include <QLineEdit>
#include <QPushButton>
#include <QLabel>
#include "ui_wizard_sb_plant_config.h"
#include "ui_wizard_sb_compute_config.h"

enum {
	Page_Intro,
	Page_Outro,
	Page_Type,
	Page_Config_Cloth,
	Page_Config_Plant,
	Page_Compute,
};

constexpr char const *fieldSrcFromScratch = "src.fromScratch";
constexpr char const *fieldSrcImportFile = "src.importFile";
constexpr char const *fieldSrcImportPath = "src.importPath";

constexpr char const *fieldExtensionNone = "ext.none";
constexpr char const *fieldExtensionCloth = "ext.cloth";
constexpr char const *fieldExtensionPlant = "ext.plant";

constexpr char const *fieldOriginX = "origin.x";
constexpr char const *fieldOriginY = "origin.y";
constexpr char const *fieldOriginZ = "origin.z";
constexpr char const *fieldPlantParticleLimit = "plant.particleLimit";
constexpr char const *fieldPlantStiffness = "plant.stiffness";
constexpr char const *fieldPlantDensity = "plant.density";
constexpr char const *fieldPlantPhotoStrength = "plant.phototropismStrength";
constexpr char const *fieldPlantAgingRate = "plant.agingRate";
constexpr char const *fieldPlantBranchingProbability = "plant.branching.probability";
constexpr char const *fieldPlantBranchingVariance = "plant.branching.variance";
constexpr char const *fieldPlantSurfaceAdaptionStrength = "plant.surface.adaption";
constexpr char const *fieldPlantSurfaceAttachmentStrength = "plant.surface.attachment";
constexpr char const *fieldComputeNone = "compute.none";
constexpr char const *fieldComputeCPU = "compute.cpu";
constexpr char const *fieldComputeOpenCL = "compute.cl";
constexpr char const *fieldComputeCUDA = "compute.cuda";

class Wizard_Page_Intro : public QWizardPage {
	Q_OBJECT;
public:
	Wizard_Page_Intro(QWidget *parent = nullptr) : QWizardPage(parent),
		_radioGroup(this),
		_layoutMain(QBoxLayout::Direction::TopToBottom, this),
		_radioScratch(this),
		_radioImport(this) {
		_radioScratch.setText("Create from scratch");
		_radioImport.setText("Import from file");
		_radioImport.setDisabled(true);
		_radioScratch.setChecked(true);
		_radioGroup.addButton(&_radioScratch);
		_radioGroup.addButton(&_radioImport);

		_btnBrowse.setText("Browse...");

		_layoutMain.addWidget(&_radioScratch);
		_layoutMain.addWidget(&_radioImport);
		_layoutMain.addWidget(&_editPath);
		_layoutMain.addWidget(&_btnBrowse);

		registerField(fieldSrcFromScratch, &_radioScratch);
		registerField(fieldSrcImportFile, &_radioImport);
		registerField(fieldSrcImportPath, &_editPath);
	}

	int nextId() const override {
		if (_radioScratch.isChecked()) {
			return Page_Type;
		} else if(_radioImport.isChecked()) {
			return Page_Compute;
		}

		assert(0);
		return -1;
	}

private:
	QButtonGroup _radioGroup;
	QBoxLayout _layoutMain;
	QRadioButton _radioScratch;
	QRadioButton _radioImport;
	QLineEdit _editPath;
	QPushButton _btnBrowse;
};

class Wizard_Page_Type : public QWizardPage {
	Q_OBJECT;
public:
	Wizard_Page_Type(QWidget *parent = nullptr) : QWizardPage(parent),
		_layout(QBoxLayout::Direction::TopToBottom),
		_lblLabel(this),
		_radioGroup(this),
		_radioNone(this),
		_radioPlant(this),
		_radioCloth(this) {
		_lblLabel.setText("Choose a simulation extension:");
		_layout.addWidget(&_lblLabel);

		_radioNone.setText("None");
		_layout.addWidget(&_radioNone);
		_radioGroup.addButton(&_radioNone);

		_radioPlant.setText("Plant");
		_radioPlant.setChecked(true);
		_layout.addWidget(&_radioPlant);
		_radioGroup.addButton(&_radioPlant);

		_radioCloth.setText("Cloth");
		_layout.addWidget(&_radioCloth);
		_radioGroup.addButton(&_radioCloth);

		setLayout(&_layout);

		registerField(fieldExtensionNone, &_radioNone);
		registerField(fieldExtensionCloth, &_radioCloth);
		registerField(fieldExtensionPlant, &_radioPlant);
	}

	int nextId() const override {
		if (_radioNone.isChecked()) {
			return Page_Outro;
		} else if (_radioCloth.isChecked()) {
			return Page_Config_Cloth;
		} else if (_radioPlant.isChecked()) {
			return Page_Config_Plant;
		}

		assert(0);
		return -1;
	}
private:
	QBoxLayout _layout;
	QLabel _lblLabel;
	QButtonGroup _radioGroup;
	QRadioButton _radioNone;
	QRadioButton _radioPlant;
	QRadioButton _radioCloth;
};

class Wizard_Page_Plant : public QWizardPage {
	Q_OBJECT;
public:
	Wizard_Page_Plant(QWidget *parent = nullptr) : QWizardPage(parent) {
		_ui.setupUi(this);

		registerField(fieldOriginX, _ui.sbOriginX);
		registerField(fieldOriginY, _ui.sbOriginY);
		registerField(fieldOriginZ, _ui.sbOriginZ);
		registerField(fieldPlantParticleLimit, _ui.sbParticleCountLimit);
		registerField(fieldPlantStiffness, _ui.sbStiffness);
		registerField(fieldPlantDensity, _ui.sbDensity);
		registerField(fieldPlantPhotoStrength, _ui.sbPhotoRespStr);
		registerField(fieldPlantAgingRate, _ui.sbAgingRate);
		registerField(fieldPlantBranchingProbability, _ui.sbBranchProb);
		registerField(fieldPlantBranchingVariance, _ui.sbBranchVar);
		registerField(fieldPlantSurfaceAdaptionStrength, _ui.sbSurfAdaptStr);
		registerField(fieldPlantSurfaceAttachmentStrength, _ui.sbSurfAttachStr);
	}

	int nextId() const override {
		return Page_Compute;
	}

	Ui::Wizard_SB_Plant_Config *ui() {
		return &_ui;
	}

private:
	Ui::Wizard_SB_Plant_Config _ui;
};

class Wizard_Page_Compute : public QWizardPage {
	Q_OBJECT;
public:
	Wizard_Page_Compute(QWidget *parent = nullptr) : QWizardPage(parent) {
		_ui.setupUi(this);
	}

	Ui::Wizard_SB_Compute_Config *ui() {
		return &_ui;
	}

private:
	Ui::Wizard_SB_Compute_Config _ui;
};

class Wizard_Page_Outro : public QWizardPage {
	Q_OBJECT;
public:
	Wizard_Page_Outro(QWidget *parent = nullptr) : QWizardPage(parent), _lblMessage(this) {
		_lblMessage.setText("Press Finish to create the softbody simulation.");
	}

	int nextId() const override {
		return -1;
	}
private:
	QLabel _lblMessage;
};