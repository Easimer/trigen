// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: Texture input widget
//

#pragma once

#include <QFileDialog>
#include <QImage>
#include <QImageReader>
#include <QLabel>
#include <QLayout>
#include <QLineEdit>
#include <QPushButton>
#include <QString>
#include <QWidget>

/**
 * A texture input widget.
 * It features a textbox containing the path to the texture file, a "browse"
 * button, a label for the name of the texture and a preview box.
 */
class QTextureWidget : public QWidget {
    Q_OBJECT;

public:
    QTextureWidget(QWidget *parent);

signals:
    /**
     * Signalled when the user changes the path to the texture file.
     * \param path Path to the texture file
     */
    void
    pathChanged(QString const &path);

protected:
    /**
     * Loads an image from file into the preview picture box.
     */
    void
    loadImage(QString const &path);

private:
    QBoxLayout _layout;
    QLabel _imageLabel;
    QImage _previewImage;
    QLineEdit _editPath;
    QPushButton _btnBrowse;

    static constexpr int IMAGE_WIDTH = 64;
    static constexpr int IMAGE_HEIGHT = 64;
};
