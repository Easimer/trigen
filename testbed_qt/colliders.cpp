// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: collider node graph
//

#include "common.h"
#include "colliders.h"
#include "raymarching.h"
#include <array>
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
#define NODE_TYPE_VEC3(name) NODE_TYPE("Vector3", name)
#define NODE_TYPE_FLOAT(name) NODE_TYPE("float", name)

template<typename Output>
class Ast_Node {
public:
    virtual ~Ast_Node() = default;

    virtual Output evaluate() = 0;
};

namespace Node {
    class Vec3 : public QtNodes::NodeData {
    public:
        ~Vec3() override = default;

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
        ~Float() override = default;

        Float() : v() {}
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
    ~Vector_Constant() override = default;

    Vector_Constant() : Vector_Constant(vec_t()) {}

    Vector_Constant(vec_t const& v) : data(std::make_shared<Node::Vec3>(v)) {
        widget = new QWidget;
        layout = new QHBoxLayout;
        widget->setLayout(layout);

        for (int i = 0; i < N; i++) {
            sb[i] = new QDoubleSpinBox(widget);
            layout->addWidget(sb[i]);
            connect(
                sb[i], (void (QDoubleSpinBox::*)(double))&QDoubleSpinBox::valueChanged,
                [&, i](double v) {
                    auto& vec = data->value();
                    if (vec[i] != v) {
                        vec[i] = v;
                        emit dataUpdated(0);
                    }
                }
            );
        }

        // Generate name
        std::array<char, 32> buf;
        auto i = snprintf(buf.data(), 31, "Vector%zu", N);
        type_name = QString((char const*)buf.data());
    }

    QString caption() const override {
        return type_name;
    }

    QString name() const override {
        return type_name;
    }

    QWidget* embeddedWidget() override {
        return widget;
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
    QWidget* widget;
    QHBoxLayout* layout;
    QDoubleSpinBox* sb[N];
};

using Vector3_Constant = Vector_Constant<3>;

class Float_Constant : public QtNodes::NodeDataModel, public Ast_Node<float> {
    Q_OBJECT;
public:
    ~Float_Constant() override = default;

    Float_Constant() : Float_Constant(0.0f) {}

    Float_Constant(float v) : data(std::make_shared<Node::Float>(v)) {
        widget = new QWidget;
        layout = new QHBoxLayout;
        widget->setLayout(layout);

        sb[0] = new QDoubleSpinBox(widget);
        layout->addWidget(sb[0]);
        connect(
            sb[0], (void (QDoubleSpinBox::*)(double))&QDoubleSpinBox::valueChanged,
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
        return widget;
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
    QHBoxLayout* layout;
    QDoubleSpinBox* sb[1];
};

class Sample_Point_Data_Source : public QtNodes::NodeDataModel, public Ast_Node<Vec3> {
    Q_OBJECT;
public:
    ~Sample_Point_Data_Source() override = default;

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
    ~Distance_Sink() override = default;

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

    QString caption() const override { return QStringLiteral("Output"); }
    QString name() const override { return QStringLiteral("Output"); }

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

/*
 * Evaluate the value of an expression in AST form. If `node` is null, then
 * return a default value.
 * @param T Expression type
 * @param node An AST node
 * @param default_value Default value
 * @return Value of expression or `default_value`
 */
template<typename T>
T evaluate_ast_or_default(Ast_Node<T>* node, T default_value) {
    if (node != NULL) {
        return node->evaluate();
    } else {
        return default_value;
    }
}

#define NODE_TO_AST_NODE(Dst, Type) Dst = dynamic_cast<Ast_Node<Type>*>(input_node)
#define RESET_AST_NODE(Dst) Dst = NULL

// Base class for collider nodes.
class Base_Collider_Data_Model : public QtNodes::NodeDataModel, public Ast_Node<float> {
    Q_OBJECT;
public:
    ~Base_Collider_Data_Model() override = default;
    Base_Collider_Data_Model() : distance(std::make_shared<Node::Float>(0)) {}

    /*
     * Gets the unique name of the collider.
     */
    virtual QString collider_name() const = 0;
    /*
     * Gets the number of input ports.
     */
    virtual unsigned input_port_count() const = 0;

    /*
     * Gets the port data type for a given input port.
     * @param idx Input port index
     */
    virtual QtNodes::NodeDataType input_port(QtNodes::PortIndex idx) const = 0;

    /*
     * Called when a node is connected to a given input port.
     * @param idx Input port index
     * @param input_node Pointer to the node that was connected.
     *
     * @note `input_node` should implement the Ast_Node<T> interface. Use this
     * fact to build the AST.
     */
    virtual void input_connected(QtNodes::PortIndex idx, QtNodes::NodeDataModel* input_node) = 0;

    /*
     * Called when a node is disconnected from a given input port.
     * @param idx Input port index
     */
    virtual void input_disconnected(QtNodes::PortIndex idx) = 0;
private:
    QString caption() const override { return collider_name(); }
    QString name() const override { return collider_name(); }

    unsigned nPorts(QtNodes::PortType type) const override {
        switch (type) {
        case QtNodes::PortType::In: return input_port_count();
        case QtNodes::PortType::Out: return 1;
        default: assert(!"Unknown port type"); return 0;
        }
    }

    QtNodes::NodeDataType dataType(QtNodes::PortType port_type, QtNodes::PortIndex idx) const override {
        switch (port_type) {
        // If the caller is asking about input ports, redirect the request to the subclass
        case QtNodes::PortType::In: return input_port(idx);
        // If the caller is asking about output ports, return the default
        case QtNodes::PortType::Out: return NODE_TYPE_FLOAT("Distance");
        default: assert(!"Unknown port type");
        }
    }

    void setInData(std::shared_ptr<QtNodes::NodeData> data, QtNodes::PortIndex portIndex) override {}

    QWidget* embeddedWidget() override { return nullptr; }

    QtNodes::NodeValidationState validationState() const override { return QtNodes::NodeValidationState::Valid; }

    std::shared_ptr<QtNodes::NodeData> outData(QtNodes::PortIndex) override { return distance; }

    void inputConnectionCreated(QtNodes::Connection const& conn) override {
        auto input_node = conn.getNode(QtNodes::PortType::Out)->nodeDataModel();
        auto idx = conn.getPortIndex(QtNodes::PortType::In);
        input_connected(idx, input_node);
    }

    void inputConnectionDeleted(QtNodes::Connection const& conn) override {
        auto idx = conn.getPortIndex(QtNodes::PortType::In);
        input_disconnected(idx);
    }

private:
    std::shared_ptr<Node::Float> distance;
};

class Base_Combination_Data_Model : public QtNodes::NodeDataModel, public Ast_Node<float> {
public:
    ~Base_Combination_Data_Model() override = default;
    Base_Combination_Data_Model() :
        ast_lhs(NULL),
        ast_rhs(NULL),
        distance(std::make_shared<Node::Float>(0)) {}
    virtual QString combination_name() const = 0;
    virtual float evaluate(float lhs, float rhs) const = 0;
private:
    QString caption() const override { return combination_name(); }
    QString name() const override { return combination_name(); }

    unsigned nPorts(QtNodes::PortType type) const override {
        switch (type) {
        case QtNodes::PortType::In: return 2;
        case QtNodes::PortType::Out: return 1;
        default: assert(!"Unknown port type"); return 0;
        }
    }

    QtNodes::NodeDataType dataType(QtNodes::PortType port_type, QtNodes::PortIndex idx) const override {
        auto lhs = QtNodes::NodeDataType NODE_TYPE_FLOAT("LHS");
        auto rhs = QtNodes::NodeDataType NODE_TYPE_FLOAT("LHS");
        switch (port_type) {
        // If the caller is asking about input ports, redirect the request to the subclass
        case QtNodes::PortType::In: return (idx == 0) ? lhs : rhs;
        // If the caller is asking about output ports, return the default
        case QtNodes::PortType::Out: return NODE_TYPE_FLOAT("Distance");
        default: assert(!"Unknown port type");
        }
    }

    void setInData(std::shared_ptr<QtNodes::NodeData> data, QtNodes::PortIndex portIndex) override {}

    QWidget* embeddedWidget() override { return nullptr; }

    QtNodes::NodeValidationState validationState() const override { return QtNodes::NodeValidationState::Valid; }

    std::shared_ptr<QtNodes::NodeData> outData(QtNodes::PortIndex) override { return distance; }

    void inputConnectionCreated(QtNodes::Connection const& conn) override {
        auto input_node = conn.getNode(QtNodes::PortType::Out)->nodeDataModel();
        auto idx = conn.getPortIndex(QtNodes::PortType::In);

        if (idx == 0) {
            NODE_TO_AST_NODE(ast_lhs, float);
        } else {
            NODE_TO_AST_NODE(ast_rhs, float);
        }
    }

    void inputConnectionDeleted(QtNodes::Connection const& conn) override {
        auto idx = conn.getPortIndex(QtNodes::PortType::In);

        if (idx == 0) {
            RESET_AST_NODE(ast_lhs);
        } else {
            RESET_AST_NODE(ast_rhs);
        }
    }

    float evaluate() override {
        return evaluate(ast_lhs->evaluate(), ast_rhs->evaluate());
    }

private:
    Ast_Node<float>* ast_lhs = NULL;
    Ast_Node<float>* ast_rhs = NULL;
    std::shared_ptr<Node::Float> distance;
};

class Union_Combination_Data_Model : public Base_Combination_Data_Model {
public:
    ~Union_Combination_Data_Model() override = default;

    QString combination_name() const override { return QStringLiteral("Sum"); }

    float evaluate(float lhs, float rhs) const override {
        return sdf::opUnion(lhs, rhs);
    }
};

class Subtract_Combination_Data_Model : public Base_Combination_Data_Model {
public:
    ~Subtract_Combination_Data_Model() override = default;

    QString combination_name() const override { return QStringLiteral("Subtract"); }

    float evaluate(float lhs, float rhs) const override {
        return sdf::opSubtract(lhs, rhs);
    }
};

class Intersect_Combination_Data_Model : public Base_Combination_Data_Model {
public:
    ~Intersect_Combination_Data_Model() override = default;

    QString combination_name() const override { return QStringLiteral("Intersect"); }

    float evaluate(float lhs, float rhs) const override {
        return sdf::opIntersect(lhs, rhs);
    }
};

class Box_Collider_Data_Model : public Base_Collider_Data_Model {
    Q_OBJECT;
public:
    ~Box_Collider_Data_Model() override = default;

    QString collider_name() const override { return QStringLiteral("Box collider"); }
    unsigned input_port_count() const override { return 2; }

    QtNodes::NodeDataType input_port(QtNodes::PortIndex idx) const override {
        switch (idx) {
        case 0: return NODE_TYPE_VEC3("Sample point");
        case 1: return NODE_TYPE_VEC3("Size");
        default: assert(!"Unknown port index");
        }
    }

    void input_connected(QtNodes::PortIndex idx, QtNodes::NodeDataModel* input_node) override {
        switch (idx) {
        case 0: NODE_TO_AST_NODE(ast_sample_point, Vec3); break;
        case 1: NODE_TO_AST_NODE(ast_size, Vec3); break;
        }
    }

    void input_disconnected(QtNodes::PortIndex idx) override {
        switch (idx) {
        case 0: RESET_AST_NODE(ast_sample_point); break;
        case 1: RESET_AST_NODE(ast_size); break;
        }
    }

    float evaluate() override {
        auto sp = evaluate_ast_or_default(ast_sample_point, {});
        auto size = evaluate_ast_or_default(ast_size, {});

        return sdf::box(size, sp);
    }
private:
    Ast_Node<Vec3>* ast_sample_point = NULL;
    Ast_Node<Vec3>* ast_size = NULL;
};

class Sphere_Collider_Data_Model : public Base_Collider_Data_Model {
    Q_OBJECT;
public:
    ~Sphere_Collider_Data_Model() override = default;

    QString collider_name() const override { return QStringLiteral("Sphere collider"); }
    unsigned input_port_count() const override { return 2; }

    QtNodes::NodeDataType input_port(QtNodes::PortIndex idx) const override {
        switch (idx) {
        case 0: return NODE_TYPE_VEC3("Sample point");
        case 1: return NODE_TYPE_FLOAT("Radius");
        default: assert(!"Unknown port index");
        }
    }

    void input_connected(QtNodes::PortIndex idx, QtNodes::NodeDataModel* input_node) override {
        switch (idx) {
        case 0: NODE_TO_AST_NODE(ast_sample_point, Vec3); break;
        case 1: NODE_TO_AST_NODE(ast_radius, float); break;
        }
    }

    void input_disconnected(QtNodes::PortIndex idx) override {
        switch (idx) {
        case 0: RESET_AST_NODE(ast_sample_point); break;
        case 1: RESET_AST_NODE(ast_radius); break;
        }
    }

    float evaluate() override {
        auto sp = evaluate_ast_or_default(ast_sample_point, {});
        auto radius = evaluate_ast_or_default(ast_radius, 0.0f);

        return sdf::sphere(radius, sp);
    }
private:
    Ast_Node<Vec3>* ast_sample_point;
    Ast_Node<float>* ast_radius;
};

inline std::shared_ptr<QtNodes::DataModelRegistry> register_data_models() {
    auto ret = std::make_shared<QtNodes::DataModelRegistry>();

    ret->registerModel<Sphere_Collider_Data_Model>("Shapes");
    ret->registerModel<Box_Collider_Data_Model>("Shapes");
    ret->registerModel<Sample_Point_Data_Source>("Sources");
    ret->registerModel<Distance_Sink>("Sinks");
    ret->registerModel<Vector3_Constant>("Constants");
    ret->registerModel<Float_Constant>("Constants");

    ret->registerModel<Union_Combination_Data_Model>("Operations");
    ret->registerModel<Subtract_Combination_Data_Model>("Operations");
    ret->registerModel<Intersect_Combination_Data_Model>("Operations");

    return ret;
}

class Collider_Builder_Widget_Impl : public Collider_Builder_Widget {
public:
    ~Collider_Builder_Widget_Impl() override = default;

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
