#include <stdio.h>
#include <stdlib.h>

#include <objscan.h>

int main(int argc, char** argv) {
    if(argc < 3) {
        printf("Usage: %s input-file.obj output-file.txt [subdivisions]\n", argv[0]);
        return EXIT_FAILURE;
    }
    
    printf("Input: %s\n", argv[1]);
    printf("Output: %s\n", argv[2]);
    
    int subdivisions;
    if(argc >= 4) {
        if(sscanf(argv[3], "%d", &subdivisions) < 1) {
            subdivisions = 32;
        }
    }
    
    objscan_extra ex;
    ex.subdivisions = subdivisions;
    
    objscan_result res;
    res.extra = &ex;
    if(!objscan_from_obj_file(&res, argv[1])) {
        printf("objscan failure!\n");
        return EXIT_FAILURE;
    }
    
    printf("Particle count: %lld\n", res.particle_count);
    printf("Connection count: %lld\n", res.connection_count);
    
    printf("Model bounding box: [%f %f %f], [%f %f %f]\n",
           ex.bb_min.x, ex.bb_min.y, ex.bb_min.z, 
           ex.bb_max.x, ex.bb_max.y, ex.bb_max.z);
    printf("Subdivisions: %f\nStep sizes: [%f %f %f]\n",
           ex.subdivisions,
           ex.step_x, ex.step_y, ex.step_z);
    printf("Threads used: %d\n", ex.threads_used);
    
    auto f = fopen(argv[2], "wb");
    if(f != NULL) {
        for(long long i = 0; i < res.particle_count; i++) {
            auto pos = res.positions[i];
            fprintf(f, "v %f %f %f\n", pos.x, pos.y, pos.z);
        }
        
        for(long long i = 0; i < res.connection_count; i++) {
            auto conn = res.connections[i];
            fprintf(f, "c %lld %lld\n", conn.idx0, conn.idx1);
        }
        fclose(f);
    }
    
    objscan_free_result(&res);
    
    return EXIT_SUCCESS;
}
