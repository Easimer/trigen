// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: meshgen window declaration
//

#pragma once

#include "common.h"
#include <memory>
#include <QWindow>
#include <QLayout>
#include "softbody.h"

QDialog *make_meshgen_window(sb::Unique_Ptr<sb::ISoftbody_Simulation> &simulation, QMainWindow *parent);
