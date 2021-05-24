// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose: converts text files to C source files
//

#include "stdafx.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <stdlib.h>

#define BINARY_FILE_ROW_WIDTH (20)

typedef void (*transcriber_fun_t)(FILE *dst, FILE *src);

// Decides whether a given character is printable.
static int is_printable(char c) {
    if(c >= ' ' && c <= '~')
        return 1;

    if (c == '\t')
        return 1;

    return 0;
}

// Decides whether we need to escape this character.
static int need_to_escape(char c) {
    switch(c) {
        case '"':
        case '\\':
            return 1;
        default:
            return 0;
    }
}

// Converts a byte to a C hex literal (\xFF) and writes it to the output stream
static void emit_hex(FILE* dst, unsigned char ch) {
    char esc[2] = { '\\', 'x' };
    char digit, c;

    fwrite(esc, 1, 2, dst);

    for(int i = 0; i < 2; i++) {
        digit = (ch >> ((1 - i) * 4)) & 0x0F;

        if(digit < 10) {
            c = '0' + digit;
        } else {
            c = 'A' + (digit - 10);
        }

        fwrite(&c, 1, 1, dst);
    }
}

// Transcribes a binary file, converting each byte from `src` into a C hex
// literal and writing it to `dst`.
static void transcribe_binary_file(FILE *dst, FILE *src) {
    assert(dst != NULL);
    assert(src != NULL);

    int column = 0;
    char byte;
    char buffer[BINARY_FILE_ROW_WIDTH];
    int res;

    // put newline after the assignment operator, so the rows are aligned
    fprintf(dst, "\n");

    while (!feof(src)) {
        res = fread(buffer, 1, BINARY_FILE_ROW_WIDTH, src);
        if (res < 1) {
            continue;
        }

        fprintf(dst, "\"");

        for (int i = 0; i < res; i++) {
            emit_hex(dst, buffer[i]);
        }

        fprintf(dst, "\"\n");
    }
}

// Transcribes a text file, copying each character from src into dst, escaping
// and/or converting to hex literals if necessary. 
static void transcribe_text_file(FILE* dst, FILE* src) {
    assert(dst != NULL);
    assert(src != NULL);

    int flag_emit_opening_quotes = 1;
    int flag_emit_closing_quotes = 1;

    while(!feof(src)) {
        char ch;
        int res;

        res = fread(&ch, 1, 1, src);

        if (res < 1) {
            continue;
        }

        if (flag_emit_opening_quotes) {
            fprintf(dst, "\"");
            flag_emit_opening_quotes = 0;
            flag_emit_closing_quotes = 1;
        }

        if(is_printable(ch)) {
            if(need_to_escape(ch)) {
                fprintf(dst, "\\");
            }
            fwrite(&ch, 1, 1, dst);
        } else {
            if (ch == '\r') {
                continue;
            }

            if(ch != '\n') {
                emit_hex(dst, ch);
            } else {
                flag_emit_closing_quotes = 0;
                fprintf(dst, "\\n\"\n");
                flag_emit_opening_quotes = 1;
            }
        }
    }

    if (flag_emit_closing_quotes) {
        fprintf(dst, "\"");
    }
}

// Decides whether a given character could be part of a C variable name.
static int can_be_part_of_variable_name(char ch) {
    return (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9');
}

// Converts variable name from filename and writes it to `dst`.
static void generate_variable_name(FILE* dst, char const* filename) {
    while(*filename != '\0') {
        char ch = *filename;
        char us = '_';

        if(can_be_part_of_variable_name(ch)) {
            fwrite(&ch, 1, 1, dst);
        } else {
            fwrite(&us, 1, 1, dst);
        }

        filename++;
    }
}

// Tries to decide whether an input file is a text file or not.
static int is_text_file(FILE *file) {
    // Nothing fancy; simply checks the file signature.
    int ret = 1;
    long orig_pos = ftell(file);

    char const *sig_png = "\x89PNG";
    // Not an actual signature but the name of the first section in
    // a compiled material blob.
    char const *sig_matc = "SREV_TAM";
    char const *sig_opentype = "\x00\x01\x00\x00";

    fseek(file, 0, SEEK_SET);
    unsigned char header[8];
    fread(header, 1, 8, file);
    if (memcmp(header, sig_png, 4) == 0) {
        fprintf(stderr, "[+] png-file-detected\n");
        goto end_binary;
    } else if (memcmp(header, sig_matc, 8) == 0) {
        fprintf(stderr, "[+] filament-material-file-detected\n");
        goto end_binary;
    } else if (memcmp(header, sig_opentype, 4) == 0) {
        fprintf(stderr, "[+] opentype-font-detected\n");
        goto end_binary;
    }

    fprintf(stderr, "[+] text-file-detected\n");
    goto end;

end_binary:
    ret = 0;
end:
    fseek(file, orig_pos, SEEK_SET);
    return ret;
}

// Generates the byte array and length variable for the input file and writes
// it to the output file.
static int generate_variables_for_input_file(FILE *dst, FILE *src, char const *filename, transcriber_fun_t transcribe) {
    assert(dst != NULL && src != NULL && filename != NULL && transcribe != NULL);

    // Print filename in a comment
    fprintf(dst, "// %s\n", filename);
    // Copy file contents into a char array
    fprintf(dst, "static char const ");
    generate_variable_name(dst, filename);
    fprintf(dst, "_data[] = ");
    transcribe(dst, src);
    fprintf(dst, ";\n");

    // Store the length of the array in a variable
    fprintf(dst, "unsigned long long ");
    generate_variable_name(dst, filename);
    fprintf(dst, "_len = sizeof(");
    generate_variable_name(dst, filename);
    fprintf(dst, "_data);\n");

    fprintf(dst, "char const *");
    generate_variable_name(dst, filename);
    fprintf(dst, " = ");
    generate_variable_name(dst, filename);
    fprintf(dst, "_data;\n\n");

    return 0;
}

// Tries to embed the input file into the output stream.
static int embed_input_file(FILE *dst, char const *path) {
    int ret = 0;
    size_t path_len = strlen(path);
    char const *filename = NULL;
    char const *cur = path + path_len;
    FILE *src;
    transcriber_fun_t transcribe = NULL;

    // Get the pointer to the file name in the path string by finding the first
    // slash character from the end of the string
    while(cur-- != path && !filename) {
        if(*cur == '/') {
            filename = cur + 1;
        }
    }

    // Didn't find any slashes, assume the path doesn't have any and it is in
    // fact the filename itself
    if(!filename) {
        filename = path;
    }

    // Open input file for reading
    fprintf(stderr, "[-] process-input-file path: '%s' filename: '%s'\n", path, filename);
    src = fopen(path, "rb");

    if(src == NULL) {
        fprintf(stderr, "[!] open-ro-fail file: '%s' reason: '%s'\n", path, strerror(errno));
        ret = -1;
        goto end;
    }

    if (is_text_file(src)) {
        transcribe = transcribe_text_file;
    } else {
        transcribe = transcribe_binary_file;
    }

    generate_variables_for_input_file(dst, src, filename, transcribe);

    fclose(src);
end:
    return ret;
}

// Processes all input paths and tries to embed them into the output stream.
static int embed_input_files(FILE *dst, int inputFileCount, char const **inputFilePaths) {
    int rc;

    fprintf(dst, "// auto-generated\n");
    fprintf(dst, "extern \"C\" {\n");

    for (int i = 0; i < inputFileCount; i++) {
        rc = embed_input_file(dst, inputFilePaths[i]);

        if (rc != 0) {
            return 1;
        }
    }

    fprintf(dst, "}");

    return 0;
}

int main(int argc, char **argv) {
    char const *output;
    FILE *dstFile;
    int rc;

    fprintf(stderr, "trigen embed built on %s\n", __DATE__);

    if(argc >= 3) {
        output = argv[1];
        dstFile = fopen(output, "wb");
        if(dstFile == NULL) {
            fprintf(stderr, "[!] open-rw-fail file: '%s' reason: '%s'\n", output, strerror(errno));
            return EXIT_FAILURE;
        }

        fprintf(stderr, "[-] output path: '%s'\n", output);

        rc = embed_input_files(dstFile, argc - 2, argv + 2);
        if (rc != 0) {
            fprintf(stderr, "[!] embed failed\n");
            unlink(output);
        }

        fclose(dstFile);
    } else {
        fprintf(stderr, "Usage: %s output-file input-file [input-file [...]]\n", argv[0]);
    }
    return 0;
}

