// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: Texture input widget
//

#include "stdafx.h"

#include <QApplication>
#include <QMessageBox>

#include "widget_texture.h"

QTextureWidget::QTextureWidget(QWidget *parent)
    : QWidget(parent)
    , _layout(QBoxLayout::Direction::LeftToRight, this)
    , _imageLabel(this)
    , _editPath(this)
    , _btnBrowse(tr("Browse..."), this) {
    _layout.addWidget(&_imageLabel);
    _layout.addWidget(&_editPath);
    _layout.addWidget(&_btnBrowse);
    connect(&_btnBrowse, &QPushButton::clicked, [&]() {
        auto path = QFileDialog::getOpenFileName(
            this, "Load a texture", QString(),
            "Images (*.png *.jpg *.bmp);;All files (*.*)");
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

void
QTextureWidget::loadImage(QString const &path) {
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