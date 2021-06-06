add_library(metadesk STATIC
    metadesk/source/md.c
    metadesk/source/md.h
)

# NOTE(danielm): load-bearing precompiled header.
# md.c itself doesn't #include md.h because you're supposed to #include md.c in
# one of your source files and since you'll #include md.h anyways they didn't
# bother #including it in their own code.
# Putting md.h in a precompiled header will make the compiler include it in
# every .c file without having to modify any of them.
target_precompile_headers(metadesk PRIVATE metadesk/source/md.h)

target_include_directories(metadesk INTERFACE metadesk/source)
