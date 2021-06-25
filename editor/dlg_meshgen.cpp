// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#include "stdafx.h"

#include "dlg_meshgen.h"
#include "vm_meshgen.h"

#include <QBoxLayout>
#include <QColorSpace>
#include <QDialog>
#include <QFileDialog>
#include <QImage>
#include <QImageReader>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPushButton>
#include <QStatusBar>

#include <ui_dlg_meshgen.h>

class QTextureWidget : public QWidget {
    Q_OBJECT;

public:
    QTextureWidget(QWidget *parent)
        : QWidget(parent)
        , _layout(QBoxLayout::Direction::LeftToRight, this)
        , _imageLabel(this)
        , _editPath(this)
        , _btnBrowse(tr("Browse..."), this) {
        _layout.addWidget(&_imageLabel);
        _layout.addWidget(&_editPath);
        _layout.addWidget(&_btnBrowse);
        connect(&_btnBrowse, &QPushButton::clicked, [&]() {
            auto path = QFileDialog::getOpenFileName(this, "Load a texture", QString(), "Images (*.png *.jpg *.bmp);;All files (*.*)");
            if (path.isEmpty()) {
                return;
            }

            _editPath.setText(path);
            loadImage(path);
            emit pathChanged(path);
        });

        _imageLabel.setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
        _imageLabel.setFixedSize(IMAGE_WIDTH, IMAGE_HEIGHT);
    }

signals:
    void pathChanged(QString const &path);

protected:
    void
    loadImage(QString const &path) {
        QImageReader reader(path);
        reader.setAutoTransform(true);
        auto const newImage = reader.read();

        if (newImage.isNull()) {
            QMessageBox::information(
                this, QGuiApplication::applicationDisplayName(),
                tr("Cannot preview texture '%1': %2")
                    .arg(QDir::toNativeSeparators(path), reader.errorString()));
            return;
        }

        _previewImage = newImage.scaled(IMAGE_WIDTH, IMAGE_HEIGHT);
        _imageLabel.setPixmap(QPixmap::fromImage(_previewImage));
        _imageLabel.adjustSize();
    }

private:
    QBoxLayout _layout;
    QLabel _imageLabel;
    QImage _previewImage;
    QLineEdit _editPath;
    QPushButton _btnBrowse;

    static constexpr int IMAGE_WIDTH = 64;
    static constexpr int IMAGE_HEIGHT = 64;
};

class Dialog_Meshgen : public Base_Dialog_Meshgen, public IMeshgen_Statusbar {
    Q_OBJECT;

public:
    Dialog_Meshgen(QWorld const *world, Entity_Handle entity, QWidget *parent)
        : Base_Dialog_Meshgen(parent)
        , _vm(world, entity, this)
        , _ui{}
        , _statusBar(this) {
        setAttribute(Qt::WA_DeleteOnClose);
        _ui.setupUi(this);

        connect(&_vm, &VM_Meshgen::exportError, [&](QString const &msg) {
            QMessageBox::critical(this, tr("Exporting has failed"), msg);
        });

        connect(&_vm, &VM_Meshgen::showExportFileDialog, [&]() {
            auto path = QFileDialog::getSaveFileName(this, "Export...", QString(), "Autodesk FBX (*.fbx)");
            if (path.isEmpty()) {
                return;
            }

            _vm.onExportPathAvailable(path);
        });

        connect(_ui.chkRenderNormals, &QCheckBox::stateChanged, &_vm, &VM_Meshgen::renderNormalsOptionChanged);

        auto btnExport = new QPushButton(tr("&Export..."));
        btnExport->setDefault(true);

        connect(btnExport, &QPushButton::clicked, &_vm, &VM_Meshgen::onExportClicked);
        connect(&_vm, &VM_Meshgen::exported, this, &QDialog::accepted);

        _ui.buttonBox->addButton(btnExport, QDialogButtonBox::ActionRole);
        connect(_ui.buttonBox, &QDialogButtonBox::rejected, this, &QDialog::rejected);

        _vm.foreachInputTexture([&](Trigen_Texture_Kind kind, char const *name, Input_Texture &tex) {
            auto texWidget = new QTextureWidget(this);
            _ui.layoutTextures->addRow(name, texWidget);
            connect(texWidget, &QTextureWidget::pathChanged, [&, kind](QString const &path) {
                pathToTextureChanged(kind, path);
            });
        });

        connect(_ui.sbMetaballRadius, qOverload<double>(&QDoubleSpinBox::valueChanged), &_vm, &VM_Meshgen::metaballRadiusChanged);
        connect(_ui.sbNumSubdivisions, qOverload<int>(&QSpinBox::valueChanged), &_vm, &VM_Meshgen::numberOfSubdivionsChanged);
        connect(_ui.sbResolution, qOverload<int>(&QSpinBox::valueChanged), &_vm, &VM_Meshgen::resolutionChanged);

        // HACKHACKHACK(danielm): trigger the valueChanged signal for fields with default values
        _vm.resolutionChanged(_ui.sbResolution->value());
        _vm.numberOfSubdivionsChanged(_ui.sbNumSubdivisions->value());
        _vm.metaballRadiusChanged(_ui.sbMetaballRadius->value());

        connect(_ui.actionInspectUV, &QAction::triggered, &_vm, &VM_Meshgen::inspectUV);

        _ui.topLayout->addWidget(&_statusBar);
    }

    ~Dialog_Meshgen() override = default;

    void onRender(gfx::Render_Queue *rq) override {
        _vm.onRender(rq);
    }

    void
    setMessage(char const *message) override {
        _statusBar.showMessage(tr(message));
    }

    void
    setBusy(bool isBusy) override {
        // TODO(danielm): spinner
        if (!isBusy) {
            _statusBar.clearMessage();
        }
    }

protected slots:
    void pathToTextureChanged(Trigen_Texture_Kind kind, QString const &path) {
        auto pathUtf8 = path.toUtf8();
        _vm.loadTextureFromPath(kind, pathUtf8.constData());
    }

private:
    VM_Meshgen _vm;
    QStatusBar _statusBar;

    Ui::Dialog_Meshgen _ui;
};

Base_Dialog_Meshgen *make_meshgen_dialog(QWorld const *world, Entity_Handle entity, QWidget *parent) {
    return new Dialog_Meshgen(world, entity, parent);
}

#include "dlg_meshgen.moc"
