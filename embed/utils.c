// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: utility function implementations
//

#include "stdafx.h"
#include "utils.h"

bool is_printable(char c) {
    if(c >= ' ' && c <= '~')
        return true;

    if (c == '\t')
        return true;

    return false;
}

bool need_to_escape(char c) {
    switch(c) {
        case '"':
        case '\\':
            return true;
        default:
            return false;
    }
}

bool can_be_part_of_variable_name(char ch) {
    return (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9');
}
