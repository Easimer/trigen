// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: common definitions
//

#pragma once

#define GLM_ENABLE_EXPERIMENTAL

#include <cassert>
#include <cstdio>

#include <memory>
#include <functional>
#include <optional>
#include <vector>
#include <unordered_map>
#include <list>

#include <glm/gtc/constants.hpp>
#include <glm/mat3x3.hpp>
#include <glm/vec3.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/matrix_decompose.hpp>

#include <QApplication>
#include <QWidget>
#include <QDebug>
#include <QVariant>
#include <QMainWindow>
#include <QTimer>
#include <QSplitter>
#include <QToolBar>
#include <QMessageBox>
#include <QOpenGLWidget>
#include <QOpenGLContext>
#include <QMouseEvent>
#include <QWheelEvent>

template<typename T>
using Unique_Ptr = std::unique_ptr<T>;

template<typename T>
using Fun = std::function<T>;

template<typename T>
using Optional = std::optional<T>;

using Vec2 = glm::vec2;
using Vec3 = glm::vec3;
using Mat3 = glm::mat3;
using Mat4 = glm::mat4;
using Quat = glm::quat;
