// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: converts text files to C source files
//

#include "stdafx.h"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <cerrno>
#include <cstdlib>
#include "main.h"

static bool is_printable(char c) {
    if(c >= ' ' && c <= '~') {
        return true;
    }

    if (c == '\t') {
        return true;
    }

    return false;
}

static bool need_to_escape(char c) {
    switch(c) {
        case '"':
        case '\\':
            return true;
        default:
            return false;
    }
}

static void emit_hex(FILE* dst, unsigned char ch) {
    char esc[2] = { '\\', 'x' };
    fwrite(esc, 1, 2, dst);

    for(int i = 0; i < 2; i++) {
        auto digit = (ch >> ((1 - i) * 4)) & 0x0F;
        char c;
        if(digit < 10) {
            c = '0' + digit;
        } else {
            c = 'A' + (digit - 10);
        }
        fwrite(&c, 1, 1, dst);
    }
}

static size_t transcribe_file(FILE* dst, FILE* src, bool always_emit_hex) {
    size_t ret = 0;
    assert(dst != NULL);
    assert(src != NULL);
    bool flag_emit_opening_quotes = true;
    bool flag_emit_closing_quotes = true;
    int col = 0;

    while(!feof(src)) {
        char ch;
        char esc = '\\';
        int res = fread(&ch, 1, 1, src);

        if (res < 1) {
            continue;
        }

        if (flag_emit_opening_quotes) {
            fwrite("\"", 1, 1, dst);
            flag_emit_opening_quotes = false;
            flag_emit_closing_quotes = true;
        }

        if(!always_emit_hex && is_printable(ch)) {
            if(need_to_escape(ch)) {
                fwrite(&esc, 1, 1, dst);
                ret++;
            }
            fwrite(&ch, 1, 1, dst);
            ret++;
        } else {
            if (ch == '\r' && !always_emit_hex) {
                continue;
            }

            if(ch != '\n' || always_emit_hex) {
                emit_hex(dst, ch);
                ret++;
            } else {
                flag_emit_closing_quotes = false;
                fwrite("\\n\"\n", 1, 4, dst);
                flag_emit_opening_quotes = true;
            }

            if (always_emit_hex) {
                col++;
                if (col == 64) {
                    flag_emit_closing_quotes = false;
                    fwrite("\"\n", 1, 2, dst);
                    flag_emit_opening_quotes = true;
                    col = 0;
                }
            }
        }
    }

    if (flag_emit_closing_quotes) {
        fwrite("\"", 1, 1, dst);
    }

    return ret;
}

static bool can_be_part_of_variable_name(char ch) {
    return (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9');
}

static void generate_variable_name(FILE* dst, char const* filename) {
    while(*filename != '\0') {
        auto ch = *filename;
        auto us = '_';

        if(can_be_part_of_variable_name(ch)) {
            fwrite(&ch, 1, 1, dst);
        } else {
            fwrite(&us, 1, 1, dst);
        }

        filename++;
    }
}

static bool is_text_file(FILE *file) {
    // Nothing fancy but only detects PNG files.
    bool ret = true;
    auto orig_pos = ftell(file);

    fseek(file, 0, SEEK_SET);
    unsigned char header[8];
    fread(header, 1, 8, file);
    if (header[0] == 0x89 && header[1] == 'P' && header[2] == 'N' && header[3] == 'G') {
        fprintf(stderr, "[+] png-file-detected\n");
        fseek(file, orig_pos, SEEK_SET);
        ret = false;
        goto end;
    }

    fprintf(stderr, "[+] text-file-detected\n");
end:
    fseek(file, orig_pos, SEEK_SET);
    return ret;
}

static bool convert_text_file(FILE *dst, FILE *src, char const *filename) {
    char const type[] = "#include <cstring>\nextern \"C\" {\nchar const* ";
    char const array[] = " = ";
    char const end_of_line[] = ";\n";
    fwrite(type, 1, strlen(type), dst);
    generate_variable_name(dst, filename);
    fwrite(array, 1, strlen(array), dst);
    transcribe_file(dst, src, false);
    fwrite(end_of_line, 1, strlen(end_of_line), dst);

    char const siz_type[] = "unsigned long long ";
    char const siz_value[] = " = strlen(";
    fwrite(siz_type, 1, strlen(siz_type), dst);
    generate_variable_name(dst, filename);
    fwrite("_len", 1, 4, dst);
    fwrite(siz_value, 1, strlen(siz_value), dst);
    generate_variable_name(dst, filename);
    fwrite(");\n}\n", 1, 5, dst);

    fclose(src);

    return true;
}

static bool convert_binary_file(FILE *dst, FILE *src, char const *filename) {
    fseek(src, 0, SEEK_END);
    auto src_len = ftell(src);
    fseek(src, 0, SEEK_SET);

    char const type[] = "extern \"C\" {\nchar const *";
    char const end_of_line[] = ";\n";

    fwrite(type, 1, strlen(type), dst);
    generate_variable_name(dst, filename);
    fprintf(dst, " = ");
    transcribe_file(dst, src, true);
    fwrite(end_of_line, 1, strlen(end_of_line), dst);

    fprintf(dst, "unsigned long long ");
    generate_variable_name(dst, filename);
    fprintf(dst, "_len = %ld;\n", src_len);
    fprintf(dst, "}\n");

    return true;
}

static bool convert_file(FILE* dst, char const* path) {
    bool ret = false;
    auto path_len = strlen(path);
    char const* filename = NULL;

    auto cur = path + path_len;

    while(cur-- != path && !filename) {
        if(*cur == '/') {
            filename = cur + 1;
        }
    }

    if(!filename) {
        filename = path;
    }

    FILE* src = fopen(path, "rb");

    if(src == NULL) {
        fprintf(stderr, "[!] open-ro-fail file: '%s' reason: '%s'\n", path, strerror(errno));
        return false;
    }

    if (is_text_file(src)) {
        ret = convert_text_file(dst, src, filename);
        goto end;
    } else {
        ret = convert_binary_file(dst, src, filename);
        goto end;
    }

end:
    fclose(src);
    return ret;

}

int main(int argc, char** argv) {
    if(argc >= 3) {
        char const* output = argv[1];
        FILE* f = fopen(output, "wb");
        if(f == NULL) {
            fprintf(stderr, "[!] open-rw-fail file: '%s' reason: '%s'\n", output, strerror(errno));
            return EXIT_FAILURE;
        }

        for(int i = 2; i < argc; i++) {
            char const* input = argv[i];
            if(!convert_file(f, input)) {
                fprintf(stderr, "[!] convert-fail file: '%s'\n", input); 
                return EXIT_FAILURE;
            }
        }

        fclose(f);
    } else {
        fprintf(stderr, "Usage: %s output-file input-file [input-file [...]]\n", argv[0]);
    }
    return 0;
}

