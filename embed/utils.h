// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: utility functions
//

#pragma once

#include <stdbool.h>

// Decides whether a given character is printable.
bool is_printable(char c);

// Decides whether we need to escape this character.
bool need_to_escape(char c);

// Decides whether a given character could be part of a C variable name.
bool can_be_part_of_variable_name(char ch);
