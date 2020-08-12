// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: collider node graph
//

#include "common.h"
#include "colliders.h"
#include "raymarching.h"
#include <nodes/NodeData>
#include <nodes/FlowScene>
#include <nodes/FlowView>
#include <nodes/ConnectionStyle>
#include <nodes/TypeConverter>
#include <nodes/DataModelRegistry>
#include <nodes/NodeDataModel>
#include <nodes/Connection>
#include <nodes/Node>
#include <QDoubleSpinBox>
#include <QLayout>

#define NODE_TYPE(id, name) { QStringLiteral(id), QStringLiteral(name) }
#define NODE_TYPE_VEC3(name) NODE_TYPE("glm::vec3", name)
#define NODE_TYPE_FLOAT(name) NODE_TYPE("float", name)

template<typename Output>
class Ast_Node {
public:
    virtual ~Ast_Node() {}

    virtual Output evaluate() = 0;
};

namespace Node {
    class Vec3 : public QtNodes::NodeData {
    public:
        virtual ~Vec3() {}

        Vec3() : v() {}
        Vec3(glm::vec3 const& other) : v(other) {}

        QtNodes::NodeDataType type() const override {
            return NODE_TYPE_VEC3("Vector3");
        }

        auto value() const { return v; }
        auto& value() { return v; }
        void set_value(glm::vec3 const& nv) { v = nv; }
    private:
        glm::vec3 v;
    };

    class Float : public QtNodes::NodeData {
    public:
        virtual ~Float() {}

        Float() : v(0.0f) {}
        Float(float other) : v(other) {}

        QtNodes::NodeDataType type() const override {
            return NODE_TYPE_FLOAT("Float");
        }

        auto value() const { return v; }
        void set_value(float nv) { v = nv; }
    private:
        float v;
    };
}

template<size_t N>
class Vector_Constant : public QtNodes::NodeDataModel, public Ast_Node<glm::vec<N, float>> {
public:
    using vec_t = glm::vec<N, float>;
    virtual ~Vector_Constant() {}

    Vector_Constant() : Vector_Constant(vec_t()) {}

    Vector_Constant(vec_t const& v) : data(std::make_shared<Node::Vec3>(v)) {
        widget.setLayout(&layout);

        for (int i = 0; i < N; i++) {
            layout.addWidget(&sb[i]);
            connect(
                &sb[i], (void (QDoubleSpinBox::*)(double))&QDoubleSpinBox::valueChanged,
                [&](double v) {
                    auto& vec = data->value();
                    if (vec[i] != v) {
                        vec[i] = v;
                        emit dataUpdated(0);
                    }
                }
            );
        }

        // Generate name
        char buf[32];
        auto i = snprintf(buf, 31, "Vector%zu", N);
        type_name = QString((char const*)buf);
    }

    QString caption() const override {
        return type_name;
    }

    QString name() const override {
        return type_name;
    }

    QWidget* embeddedWidget() override {
        return &widget;
    }

    unsigned int nPorts(QtNodes::PortType portType) const override {
        if (portType == QtNodes::PortType::Out) return 1;
        return 0;
    }

    QtNodes::NodeDataType dataType(QtNodes::PortType portType, QtNodes::PortIndex portIndex) const override {
        return { type_name, type_name };
    }

    std::shared_ptr<QtNodes::NodeData> outData(QtNodes::PortIndex) override {
        return data;
    }

    void setInData(std::shared_ptr<QtNodes::NodeData>, int) override {}

    glm::vec3 evaluate() override {
        return data->value();
    }
private:
    QString type_name;
    std::shared_ptr<Node::Vec3> data;
    QWidget widget;
    QHBoxLayout layout;
    QDoubleSpinBox sb[N];
};

using Vector3_Constant = Vector_Constant<3>;

class Float_Constant : public QtNodes::NodeDataModel, public Ast_Node<float> {
    Q_OBJECT;
public:
    virtual ~Float_Constant() {}

    Float_Constant() : Float_Constant(0.0f) {}

    Float_Constant(float v) : data(std::make_shared<Node::Float>(v)) {
        widget.setLayout(&layout);

        layout.addWidget(&sb[0]);
        connect(
            &sb[0], (void (QDoubleSpinBox::*)(double))&QDoubleSpinBox::valueChanged,
            [&](double v) {
                if (data->value() != v) {
                    data->set_value(v);
                    emit dataUpdated(0);
                }
            }
        );
    }

    QString caption() const override {
        return QStringLiteral("Float");
    }

    QString name() const override {
        return QStringLiteral("Float constant");
    }

    QWidget* embeddedWidget() override {
        return &widget;
    }

    unsigned int nPorts(QtNodes::PortType portType) const override {
        if (portType == QtNodes::PortType::Out) return 1;
        return 0;
    }

    QtNodes::NodeDataType dataType(QtNodes::PortType portType, QtNodes::PortIndex portIndex) const override {
        return NODE_TYPE_FLOAT("Value");
    }

    std::shared_ptr<QtNodes::NodeData> outData(QtNodes::PortIndex) override {
        return data;
    }

    void setInData(std::shared_ptr<QtNodes::NodeData>, int) override {}

    float evaluate() override {
        return data->value();
    }

private:
    std::shared_ptr<Node::Float> data;
    QWidget* widget;
    QHBoxLayout layout;
    QDoubleSpinBox sb[1];
};

class Sample_Point_Data_Source : public QtNodes::NodeDataModel, public Ast_Node<Vec3> {
    Q_OBJECT;
public:
    virtual ~Sample_Point_Data_Source() {}

    Sample_Point_Data_Source() {
        data = std::make_shared<Node::Vec3>(glm::vec3());
    }

    void setValue(glm::vec3 const& v) {
        value = v;
        // emit dataUpdated(0);
    }

    glm::vec3 evaluate() override {
        return value;
    }

protected:
    void setInData(std::shared_ptr<QtNodes::NodeData> data, int) override {}

    QString caption() const override {
        return QStringLiteral("Sample point");
    }

    QString name() const override {
        return QStringLiteral("Sample point");
    }

    QWidget* embeddedWidget() override { return nullptr; }

    unsigned int nPorts(QtNodes::PortType portType) const override {
        if (portType == QtNodes::PortType::Out) return 1;
        return 0;
    }

    QtNodes::NodeDataType dataType(QtNodes::PortType portType, QtNodes::PortIndex portIndex) const override {
        return NODE_TYPE_VEC3("Value");
    }

    std::shared_ptr<QtNodes::NodeData> outData(QtNodes::PortIndex) override {
        return data;
    }

private:
    // decoy
    std::shared_ptr<Node::Vec3> data;
    // this is the actual value
    glm::vec3 value;
};

class Distance_Sink : public QtNodes::NodeDataModel, public Ast_Node<float> {
    Q_OBJECT;
public:
    virtual ~Distance_Sink() {}

    void setInData(std::shared_ptr<QtNodes::NodeData> data, int) override {
        this->data = std::static_pointer_cast<Node::Float>(data);
    }

    QtNodes::NodeDataType dataType(QtNodes::PortType portType, QtNodes::PortIndex portIndex) const override {
        return NODE_TYPE_FLOAT("Output");
    }

    std::shared_ptr<QtNodes::NodeData> outData(QtNodes::PortIndex) override {
        return nullptr;
    }

    float value() const {
        auto v = data.lock();
        if(v != nullptr) {
            return v->value();
        } else {
            return INFINITY;
        }
    }

    QString caption() const override {
        return QStringLiteral("Output");
    }

    QString name() const override {
        return QStringLiteral("Output");
    }

    unsigned int nPorts(QtNodes::PortType type) const override {
        using QtNodes::PortType;
        switch (type) {
        case PortType::In: return 1;
        case PortType::Out: return 0;
        default: assert(!"Unknown port type"); return 0;
        }
    }

    QWidget* embeddedWidget() override { return nullptr; }

    void inputConnectionCreated(QtNodes::Connection const& conn) override {
        auto input_node = conn.getNode(QtNodes::PortType::Out)->nodeDataModel();
        ast_distance = dynamic_cast<Ast_Node<float>*>(input_node);
    }

    void inputConnectionDeleted(QtNodes::Connection const&) override {
        ast_distance = NULL;
    }

    float evaluate() override {
        if (ast_distance) {
            return ast_distance->evaluate();
        } else {
            return NAN;
        }
    }

private:
    std::weak_ptr<Node::Float> data;

    Ast_Node<float>* ast_distance;
};

class Sphere_Collider_Data_Model : public QtNodes::NodeDataModel, public Ast_Node<float> {
    Q_OBJECT;
public:
    virtual ~Sphere_Collider_Data_Model() {}

    Sphere_Collider_Data_Model() {
        distance = std::make_shared<Node::Float>(INFINITY);
    }

    QString caption() const override {
        return QStringLiteral("Sphere Collider");
    }

    QString name() const override {
        return QStringLiteral("Sphere Collider");
    }

    unsigned int nPorts(QtNodes::PortType type) const override {
        using QtNodes::PortType;
        switch (type) {
        case PortType::In: return 2;
        case PortType::Out: return 1;
        default: assert(!"Unknown port type"); return 0;
        }
    }

    QtNodes::NodeDataType dataType(QtNodes::PortType port_type, QtNodes::PortIndex idx) const override {
        using QtNodes::PortType;
        switch (port_type) {
        case PortType::In:
        {
            switch (idx) {
            case 0: return NODE_TYPE_VEC3("Sample point");
            case 1: return NODE_TYPE_FLOAT("Radius");
            default: assert(!"Unknown port index");
            }
            break;
        }
        case PortType::Out:
        {
            return NODE_TYPE_FLOAT("Distance");
        }
        default:
        {
            assert(!"Unknown port type");
        }
        }
    }

    void setInData(std::shared_ptr<QtNodes::NodeData> data, QtNodes::PortIndex portIndex) override {
        switch (portIndex) {
        case 0:
        {
            sample_point = std::static_pointer_cast<Node::Vec3, QtNodes::NodeData>(data);
            break;
        }
        case 1:
        {
            radius = std::static_pointer_cast<Node::Float, QtNodes::NodeData>(data);
            break;
        }
        }

        auto r = radius.lock();
        auto sp = sample_point.lock();
        if (r != nullptr && sp != nullptr) {
            distance->set_value(sdf::sphere(r->value(), sp->value()));

            emit dataUpdated(0);
        }
    }

    QWidget* embeddedWidget() override { return nullptr; }

    QtNodes::NodeValidationState validationState() const override {
        if (!sample_point.expired() && !radius.expired()) {
            auto r = radius.lock();
            if (r->value() > 0.0f) {
                return QtNodes::NodeValidationState::Valid;
            } else {
                return QtNodes::NodeValidationState::Warning;
            }
        } else {
            return QtNodes::NodeValidationState::Error;
        }
    }

    std::shared_ptr<QtNodes::NodeData> outData(QtNodes::PortIndex) override {
        return distance;
    }

    void inputConnectionCreated(QtNodes::Connection const& conn) override {
        auto input_node = conn.getNode(QtNodes::PortType::Out)->nodeDataModel();
        auto idx = conn.getPortIndex(QtNodes::PortType::In);
        switch (idx) {
        case 0:
        {
            ast_sample_point = dynamic_cast<Ast_Node<Vec3>*>(input_node);
            break;
        }
        case 1:
        {
            ast_radius = dynamic_cast<Ast_Node<float>*>(input_node);
            break;
        }
        }
    }

    void inputConnectionDeleted(QtNodes::Connection const& conn) override {
        auto idx = conn.getPortIndex(QtNodes::PortType::In);
        switch (idx) {
        case 0:
        {
            ast_sample_point = NULL;
            break;
        }
        case 1:
        {
            ast_radius = NULL;
            break;
        }
        }
    }

    float evaluate() override {
        float radius = 0;
        Vec3 sp = {};
        if (ast_radius != NULL) {
            radius = ast_radius->evaluate();
        }
        if (ast_sample_point != NULL) {
            sp = ast_sample_point->evaluate();
        }

        return sdf::sphere(radius, sp);
    }
private:
    std::weak_ptr<Node::Vec3> sample_point;
    std::weak_ptr<Node::Float> radius;

    Ast_Node<Vec3>* ast_sample_point;
    Ast_Node<float>* ast_radius;

    std::shared_ptr<Node::Float> distance;
};

inline std::shared_ptr<QtNodes::DataModelRegistry> register_data_models() {
    auto ret = std::make_shared<QtNodes::DataModelRegistry>();

    ret->registerModel<Sphere_Collider_Data_Model>("Shape");
    ret->registerModel<Sample_Point_Data_Source>("Sources");
    ret->registerModel<Distance_Sink>("Sinks");
    ret->registerModel<Vector3_Constant>("Constants");
    ret->registerModel<Float_Constant>("Constants");

    return ret;
}

class Collider_Builder_Widget_Impl : public Collider_Builder_Widget {
public:
    virtual ~Collider_Builder_Widget_Impl() {}

    Collider_Builder_Widget_Impl(QWidget* parent = nullptr) :
    collider_scene(register_data_models(), parent),
    collider_view(&collider_scene) {
        auto sp = std::make_unique<Sample_Point_Data_Source>();
        auto d = std::make_unique<Distance_Sink>();

        input = sp.get();
        output = d.get();

        collider_scene.createNode(std::move(sp));
        collider_scene.createNode(std::move(d));
    }

    QWidget* view() override {
        return &collider_view;
    }

    float evaluate(glm::vec3 const& sample_point) override {
        input->setValue(sample_point);
        return output->evaluate();
    }

private:
    QtNodes::FlowScene collider_scene;
    QtNodes::FlowView collider_view;

    Sample_Point_Data_Source* input;
    Distance_Sink* output;
};

std::unique_ptr<Collider_Builder_Widget> create_collider_builder_widget(QWidget* parent) {
    return std::make_unique<Collider_Builder_Widget_Impl>(parent);
}

#include "colliders.moc"
