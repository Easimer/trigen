cmake_minimum_required(VERSION 3.16)
project(trigen)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup(TARGETS)

# =======================================================
# Purpose: Set the target's build directory
# =======================================================
macro(tri_builddir target)
	# For executables
	set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/Debug")
	set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/Release")
	set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_BINARY_DIR}/RelWithDebInfo")
	# For libraries
	set_target_properties(${target} PROPERTIES LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/Debug/bin")
	set_target_properties(${target} PROPERTIES LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/Release/bin")
	set_target_properties(${target} PROPERTIES LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_BINARY_DIR}/RelWithDebInfo/bin")
endmacro()

# =======================================================
# Purpose: Defines the TRIGEN_DLLEXPORT and TRIGEN_DLLIMPORT macros that
# expand to 
# - __declspec(dllexport) and __declspec(dllimport) on MSVC
# - __attribute__((dllexport)) and __attribute__((dllimport)) everywhere else
# and on every platform except MSVC it adds `-fvisibility=hidden` to the
# target's flags property.
# =======================================================
macro(tri_dllimportexport target)
	if(MSVC)
		target_compile_definitions(${target}
			PUBLIC	
				TRIGEN_DLLEXPORT=__declspec\(dllexport\)
				TRIGEN_DLLIMPORT=__declspec\(dllimport\)
		)
	else()
		target_compile_definitions(${target}
			PUBLIC
				TRIGEN_DLLEXPORT=__attribute__\(\(visibility\(\"default\"\)\)\)
				TRIGEN_DLLIMPORT=
		)
		set_target_properties(${target} PROPERTIES CXX_VISIBILITY_PRESET hidden)
	endif()
endmacro()

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 52 61 75)
endif()

option(TRIGEN_BUILD_CSHARP_BINDING "Build the C# bindings for libtrigen")
if(${TRIGEN_BUILD_CSHARP_BINDING})
	enable_language(CSharp)
endif()

# Include public headers for all targets
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/public")

# Use folders
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

enable_testing()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

option(SOFTBODY_CLANG_TIDY "Run clang-tidy")

if(${SOFTBODY_CLANG_TIDY})
    if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
        message(FATAL_ERROR "SOFTBODY_CLANG_TIDY requires an out-of-source build!")
    endif()
    find_program(CLANG_TIDY NAMES clang-tidy)
    if(NOT CLANG_TIDY)
        message(WARNING "clang-tidy was not found!")
        set(CMAKE_CXX_CLANG_TIDY "" CACHE STRING "" FORCE)
    else()
        set(CLANG_TIDY_CHECKS "-*,performance-*,readability-*,portability-*,bugprone-*,hicpp-*,cert-*,modernize-*,-modernize-use-trailing-return-type,-readability-magic-numbers,-hicpp-uppercase-literal-suffix,-readability-uppercase-literal-suffix")
        set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY};-checks=${CLANG_TIDY_CHECKS}")
		message("Clang-Tidy enabled!")
    endif()
endif()

macro(target_disable_clang_tidy TGT)
	set_target_properties(${TGT} PROPERTIES C_CLANG_TIDY "" CXX_CLANG_TIDY "")
endmacro()

find_package(SDL2 REQUIRED)
find_package(SDL2_ttf REQUIRED)
find_package(OpenCL REQUIRED)

set(Boost_INSTALL_DIR ${CONAN_BOOST_ROOT})
find_package(Boost REQUIRED COMPONENTS system thread)

find_package(Tracy)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

add_subdirectory(embed)

add_subdirectory(libs)
add_subdirectory(renderer)
add_subdirectory(objscan)

add_subdirectory(softbody_crosscheck)

add_subdirectory(mesh_paint_proto)
add_subdirectory(editor)

option(SOFTBODY_TESTBED_QT "Build the Qt5-based testbed")

if(${SOFTBODY_TESTBED_QT})
message("Will build Qt testbed")
add_subdirectory(nodeeditor)
# NOTE(danielm): nodeeditor doesn't seem to work with unity builds
set_target_properties(nodes PROPERTIES UNITY_BUILD OFF)
target_disable_clang_tidy(nodes)
set_target_properties(nodes PROPERTIES FOLDER "Libs")
# Precompile some headers for nodeeditor.
target_precompile_headers(
	nodes PRIVATE
	<QApplication>
	<QDebug>
	<QPainter>
	<QtCore/QBuffer>
	<QtCore/QByteArray>
	<QtCore/QDataStream>
	<QtCore/QDebug>
	<QtCore/QDir>
	<QtCore/QEvent>
	<QtCore/QFile>
	<QtCore/QJsonArray>
	<QtCore/QJsonDocument>
	<QtCore/QJsonObject>
	<QtCore/QJsonValue>
	<QtCore/QJsonValueRef>
	<QtCore/QMargins>
	<QtCore/QObject>
	<QtCore/QPoint>
	<QtCore/QPointF>
	<QtCore/QRectF>
	<QtCore/QString>
	<QtCore/QUuid>
	<QtCore/QVariant>
	<QtCore/QVariantMap>
	<QtCore/QtGlobal>
	<QtGlobal>
	<QtGui/QBrush>
	<QtGui/QColor>
	<QtGui/QDoubleValidator>
	<QtGui/QFontMetrics>
	<QtGui/QIcon>
	<QtGui/QPainter>
	<QtGui/QPen>
	<QtGui/QPixmap>
	<QtGui/QTransform>
	<QtOpenGL>
	<QtWidgets/QApplication>
	<QtWidgets/QFileDialog>
	<QtWidgets/QGraphicsBlurEffect>
	<QtWidgets/QGraphicsDropShadowEffect>
	<QtWidgets/QGraphicsEffect>
	<QtWidgets/QGraphicsItem>
	<QtWidgets/QGraphicsObject>
	<QtWidgets/QGraphicsScene>
	<QtWidgets/QGraphicsSceneMouseEvent>
	<QtWidgets/QGraphicsSceneMoveEvent>
	<QtWidgets/QGraphicsView>
	<QtWidgets/QLabel>
	<QtWidgets/QLineEdit>
	<QtWidgets/QMenu>
	<QtWidgets/QMenuBar>
	<QtWidgets/QMessageBox>
	<QtWidgets/QStyleOptionGraphicsItem>
	<QtWidgets/QVBoxLayout>
	<QtWidgets/QWidget>
	<QtWidgets/QtWidgets>
	<QtWidgets>
	<catch2/catch.hpp>
	<cmath>
	<cstdlib>
	<functional>
	<iostream>
	<memory>
	<nodes/Connection>
	<nodes/ConnectionStyle>
	<nodes/DataModelRegistry>
	<nodes/FlowScene>
	<nodes/FlowView>
	<nodes/FlowViewStyle>
	<nodes/Node>
	<nodes/NodeData>
	<nodes/NodeDataModel>
	<nodes/NodeStyle>
	<nodes/TypeConverter>
	<set>
	<stdexcept>
	<tuple>
	<unordered_map>
	<utility>
	<vector>
)

add_subdirectory(testbed_qt)
endif()
