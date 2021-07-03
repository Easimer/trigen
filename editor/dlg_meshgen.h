// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: Mesh generation dialog
//

#pragma once

#include <memory>
#include <QDialog>
#include <r_queue.h>

#include "world_qt.h"

/**
 * Base class for the mesh generation dialog
 */
class Base_Dialog_Meshgen : public QDialog {
    Q_OBJECT;

public:
    Base_Dialog_Meshgen(QWidget *parent = nullptr)
        : QDialog(parent) {
    }

    virtual ~Base_Dialog_Meshgen() = default;

public slots:
    /**
     * Renders the generated mesh
     * \param[inout] rq Render queue
     */
    virtual void onRender(gfx::Render_Queue *rq) = 0;
    /**
     * Renders the transparent parts of the generated mesh
     * \param[inout] rq Render queue
     */
    virtual void onRenderTransparent(gfx::Render_Queue *rq) = 0;
};

/**
 * Instantiates a new mesh generation dialog for a specific entity.
 * That entity lives in the `world` world and must have a plant component.
 * 
 * \param world The world
 * \param entity Handle to the plant entity
 * \param parent Parent of the dialog
 * \return Pointer to the dialog
 */
Base_Dialog_Meshgen *make_meshgen_dialog(QWorld const *world, Entity_Handle entity, QWidget *parent);
