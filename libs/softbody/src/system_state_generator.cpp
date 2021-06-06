// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#include <cassert>
#include <cstdio>
#include <cstdlib>

#include <memory>
#include <vector>

// md.h must be included before md.c
#include <md.h>

#include <md.c>

#ifdef _WIN32
#include <direct.h> // getcwd
#else
#include <unistd.h> // getcwd
#endif

struct Attribute {
    bool doc = false;
    bool map_1_to_N = false;
    bool nonattribute = false;
    bool uniform = false;
    bool set = false;

    MD_String8 name;
    MD_String8 type;

    MD_Node *nodeDoc = nullptr;

    MD_Node *nodeMapTo = nullptr;

    MD_Node *nodeSetElementType = nullptr;
};

static void print_error(char const *fmt, ...) {
    va_list va;
    va_start(va, fmt);
    vfprintf(stderr, fmt, va);
    va_end(va);
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
                    print_error("ERROR: attribute '%.*s' has tag 'set' with no type specified\n", nodeAttribute->string);
                    std::abort();
                }
            } else if (MD_StringMatch(nodeTag->string, MD_S8CString("map"), MD_MatchFlag_CaseInsensitive)) {
                if (nodeTag->first_child != nullptr && nodeTag->first_child->next != nullptr) {
                    attr.map_1_to_N = true;
                    attr.nodeMapTo = nodeTag->first_child;
                } else {
                    print_error("ERROR: attribute '%.*s' has tag 'map_1_to_N' with some types missing\n", nodeAttribute->string);
                    std::abort();
                }
            }
        }

        ret.push_back(attr);
    }

    return ret;
}

static void generate_struct_member(FILE *out, Attribute const &attr) {
    if (attr.doc) {
        assert(attr.nodeDoc->string.str != nullptr);
        fprintf(out, "    /** %.*s */\n", MD_StringExpand(attr.nodeDoc->string));
    }

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

static void emit(FILE *out, MD_ParseResult &parse) {
    fprintf(out, "// </auto-generated>\n\n");
    for (MD_EachNode(node, parse.node->first_child)) {
        auto attributes = process_attributes(node);
        generate_structs(out, node, attributes);
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
