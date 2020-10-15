#include <stdio.h>
#include <stdlib.h>

#include <objscan.h>

int main(int argc, char** argv) {
    if(argc < 3) {
        printf("Usage: %s input-file.obj output-file.txt\n", argv[0]);
        return EXIT_FAILURE;
    }
    
    printf("Input: %s\n", argv[1]);
    printf("Output: %s\n", argv[2]);
    
    objscan_result res;
    if(!objscan_from_obj_file(&res, argv[1])) {
        printf("objscan failure!\n");
        return EXIT_FAILURE;
    }
    
    printf("Particle count: %lld\n", res.particle_count);
    
    auto f = fopen(argv[2], "wb");
    if(f != NULL) {
        for(long long i = 0; i < res.particle_count; i++) {
            auto pos = res.positions[i];
            fprintf(f, "%f %f %f\n", pos.x, pos.y, pos.z);
        }
        fclose(f);
    }
    
    objscan_free_result(&res);
    
    return EXIT_SUCCESS;
}
