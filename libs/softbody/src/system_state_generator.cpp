// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#include <cassert>
#include <cstdio>
#include <cstdlib>

#include <memory>
#include <vector>

#pragma clang diagnostic ignored "-Wwritable-strings"

// md.h must be included before md.c
#include <md.h>

#include <md.c>

#ifdef _WIN32
#include <direct.h> // getcwd
#else
#include <unistd.h> // getcwd
#endif

enum class Type {
    Vec4,
    Mat4,
    Float,
    Quat,

    Other
};

struct Attribute {
    bool doc = false;
    bool map_1_to_N = false;
    bool nonattribute = false;
    bool uniform = false;
    bool set = false;
    bool noinit = false;

    MD_String8 name;
    MD_String8 type;

    Type kType = Type::Other;

    MD_Node *node = nullptr;

    MD_Node *nodeDoc = nullptr;

    MD_Node *nodeMapTo = nullptr;

    MD_Node *nodeSetElementType = nullptr;

    bool is_normal_attribute() const noexcept {
        return !map_1_to_N && !nonattribute && !uniform && !set;
    }

    bool is_map() const noexcept {
        return map_1_to_N;
    }

    bool is_type(char const *cstring) const {
        return MD_StringMatch(type, MD_S8CString((char*)cstring), 0);
    }
};

static void print_base(char const *tag, char const *fmt, va_list va) {
    fprintf(stderr, "%s: ", tag);
    vfprintf(stderr, fmt, va);
}

static void print_error(char const *fmt, ...) {
    va_list va;
    va_start(va, fmt);
    print_base("ERROR", fmt, va);
    va_end(va);
}

static void print_warning(char const *fmt, ...) {
    va_list va;
    va_start(va, fmt);
    print_base("WARNING", fmt, va);
    va_end(va);
}

static void print_note(char const *fmt, ...) {
    va_list va;
    va_start(va, fmt);
    print_base("NOTE", fmt, va);
    va_end(va);
}


static void cache_type(Attribute &attr) {
    if (attr.is_type("Vec4")) {
        attr.kType = Type::Vec4;
    } else if(attr.is_type("Mat4")) {
        attr.kType = Type::Mat4;
    } else if(attr.is_type("Quat")) {
        attr.kType = Type::Quat;
    } else if(attr.is_type("float")) {
        attr.kType = Type::Float;
    } else {
        attr.kType = Type::Other;
    }
}

static std::vector<Attribute> process_attributes(MD_Node *nodeStruct) {
    std::vector<Attribute> ret;

    for (MD_EachNode(nodeAttribute, nodeStruct->first_child)) {
        Attribute attr;

        attr.name = nodeAttribute->string;
        if(nodeAttribute->first_child != nullptr) {
            attr.type = nodeAttribute->first_child->string;
        }

        for (MD_EachNode(nodeTag, nodeAttribute->first_tag)) {
            if (MD_StringMatch(nodeTag->string, MD_S8CString("doc"), MD_MatchFlag_CaseInsensitive)) {
                if (nodeTag->first_child != nullptr && nodeTag->first_child->string.str != nullptr) {
                    attr.doc = true;
                    attr.nodeDoc = nodeTag->first_child;
                }
            } else if (MD_StringMatch(nodeTag->string, MD_S8CString("nonattribute"), MD_MatchFlag_CaseInsensitive)) {
                attr.nonattribute = true;
            } else if (MD_StringMatch(nodeTag->string, MD_S8CString("uniform"), MD_MatchFlag_CaseInsensitive)) {
                attr.uniform = true;
            } else if (MD_StringMatch(nodeTag->string, MD_S8CString("set"), MD_MatchFlag_CaseInsensitive)) {
                if (nodeTag->first_child != nullptr && nodeTag->first_child->string.str != nullptr) {
                    attr.set = true;
                    attr.nodeSetElementType = nodeTag->first_child;
                } else {
                    print_error("attribute '%.*s' has tag 'set' with no type specified\n", nodeAttribute->string);
                    std::abort();
                }
            } else if (MD_StringMatch(nodeTag->string, MD_S8CString("map"), MD_MatchFlag_CaseInsensitive)) {
                if (nodeTag->first_child != nullptr && nodeTag->first_child->next != nullptr) {
                    attr.map_1_to_N = true;
                    attr.nodeMapTo = nodeTag->first_child;
                } else {
                    print_error("attribute '%.*s' has tag 'map_1_to_N' with some types missing\n", nodeAttribute->string);
                    std::abort();
                }
            } else if (MD_StringMatch(nodeTag->string, MD_S8CString("noinit"), MD_MatchFlag_CaseInsensitive)) {
                attr.noinit = true;
            }
        }

        attr.node = nodeAttribute;
        cache_type(attr);
        ret.push_back(attr);
    }

    return ret;
}

static void generate_doxygen_docs(FILE *out, Attribute const &attr) {
    fprintf(out, "    /**\n");
    
    if (attr.doc) {
        assert(attr.nodeDoc->string.str != nullptr);
        fprintf(out, "    * \\brief %.*s\n    *\n", MD_StringExpand(attr.nodeDoc->string));
    }

    if(attr.node->first_tag != nullptr && attr.node->first_tag->string.str != nullptr) {
        fprintf(out, "    * \\remark Tagged in the original Metadesk file:\n");
        for (MD_EachNode(nodeTag, attr.node->first_tag)) {
            fprintf(out, "    * - %.*s\n", MD_StringExpand(nodeTag->string));
        }
        fprintf(out, "    *\n");
    }

    fprintf(out, "    */\n");
}

static void generate_struct_member(FILE *out, Attribute const &attr) {
    generate_doxygen_docs(out, attr);

    assert(attr.name.str != nullptr);

    if (attr.uniform) {
        assert(attr.type.str != nullptr);
        assert(attr.name.str != nullptr);
        fprintf(out, "    %.*s %.*s;\n", MD_StringExpand(attr.type), MD_StringExpand(attr.name));
    } else if (attr.set) {
        assert(attr.name.str != nullptr);
        assert(attr.nodeSetElementType != nullptr);
        fprintf(out, "    Set<%.*s> %.*s;\n", MD_StringExpand(attr.nodeSetElementType->string), MD_StringExpand(attr.name));
    } else if (attr.map_1_to_N) {
        assert(attr.name.str != nullptr);
        assert(attr.nodeMapTo != nullptr);
        fprintf(out, "    Map<index_t, %.*s> %.*s;\n", MD_StringExpand(attr.nodeMapTo->string), MD_StringExpand(attr.name));
    } else {
        assert(attr.type.str != nullptr);
        assert(attr.name.str != nullptr);
        fprintf(out, "    Vector<%.*s> %.*s;\n", MD_StringExpand(attr.type), MD_StringExpand(attr.name));
    }
}

static void generate_structs(FILE *out, MD_Node *node, std::vector<Attribute> const &attributes) {
    fprintf(out, "struct %.*s {\n", MD_StringExpand(node->string));

    for (auto &attribute : attributes) {
        generate_struct_member(out, attribute);
    }

    fprintf(out, "};\n\n");
}

static void generate_index_calculation(FILE *out, MD_Node *nodeStruct, std::vector<Attribute> const &attributes) {
    Attribute const *attr = nullptr;
    for(auto &attribute : attributes) {
        if(attribute.is_normal_attribute()) {
            attr = &attribute;
            break;
        }
    }

    if (attr == nullptr) {
        print_warning("struct '%.*s' doesn't have any non-uniform attributes!\n", MD_StringExpand(nodeStruct->string));
        return;
    }

    fprintf(out, "    index_t const _index = s.%.*s.size();\n", MD_StringExpand(attr->name));
}

static void generate_element_creator(FILE *out, MD_Node *nodeStruct, std::vector<Attribute> const &attributes) {
    fprintf(out, "inline index_t create_element(%.*s &s) {\n", MD_StringExpand(nodeStruct->string));
    
    generate_index_calculation(out, nodeStruct, attributes);

    for (auto &attr : attributes) {
        if (attr.noinit) {
            continue;
        }

        if (!attr.nonattribute) {
            if (attr.is_normal_attribute()) {
                if(attr.kType == Type::Other) {
                    print_note("Normal attribute '%.*s' has unknown type '%.*s' and won't be initialized!", MD_StringExpand(attr.name), MD_StringExpand(attr.type));
                    continue;
                }
                fprintf(out, "    s.%.*s.emplace_back(", MD_StringExpand(attr.name));
                switch (attr.kType) {
                    case Type::Vec4:
                    fprintf(out, "Vec4(0.0f, 0.0f, 0.0f, 0.0f)");
                    break;
                    case Type::Mat4:
                    fprintf(out, "Mat4(1.0f)");
                    break;
                    case Type::Quat:
                    fprintf(out, "Quat(1.0f, 0.0f, 0.0f, 0.0f)");
                    break;
                    case Type::Float:
                    fprintf(out, "0.0f");
                    break;
                }
                fprintf(out, ");\n");
            } else if (attr.is_map()) {
                fprintf(out, "    s.%.*s[_index] = {};\n", MD_StringExpand(attr.name));
            }
        }
    }

    fprintf(out, "    return _index;\n");
    fprintf(out, "}\n\n");
}

static void emit(FILE *out, MD_ParseResult &parse) {
    fprintf(out, "// </auto-generated>\n\n");
    for (MD_EachNode(node, parse.node->first_child)) {
        auto attributes = process_attributes(node);
        generate_structs(out, node, attributes);
        generate_element_creator(out, node, attributes);
    }
}

int main(int argc, char **argv) {
    fprintf(stderr, "system_state_generator\n");

    if (argc < 3) {
        fprintf(stderr, "ERROR: Not enough arguments!\n");
        return 1;
    }

    auto output_path = argv[1];
    auto input_path = argv[2];

    char cwdbuf[512];
    getcwd(cwdbuf, 512);
    fprintf(stderr, "Working directory: %s\nOutput path: %s\nInput path: %s\n", cwdbuf, output_path, input_path);

    FILE *out = fopen(output_path, "wb");
    if(out == nullptr) {
        fprintf(stderr, "ERROR: Couldn't open output file for writing!\n");
        return 1;
    }

    auto parse = MD_ParseWholeFile(MD_S8CString(input_path));
    emit(out, parse);
    fclose(out);

    return 0;
}
