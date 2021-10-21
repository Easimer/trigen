// === Copyright (c) 2020-2021 easimer.net. All rights reserved. ===
//
// Purpose:
//

#pragma once

#include <cuda.h>
#include "cuda_utils.cuh"

struct Particle_Correction_Info {
    float4 pos_bind; // bind position wrt center of mass
    float inv_num_clusters; // 1 / number of clusters
};

using Internal_Force_Buffer = CUDA_Array<float4, struct Internal_Force_Buffer_Tag>;
using Adjacency_Table_Buffer = CUDA_Array<unsigned, struct Adjacency_Table_Buffer_Tag>;
using Cluster_Matrix_Buffer = CUDA_Array<float4, struct Cluster_Matrix_Buffer_Tag>;
using Position_Buffer = CUDA_Array<float4, struct Position_Buffer_Tag>;
using Rotation_Buffer = CUDA_Array<float4, struct Rotation_Buffer_Tag>;
using Velocity_Buffer = CUDA_Array<float4, struct Velocity_Buffer_Tag>;
using Angular_Velocity_Buffer = CUDA_Array<float4, struct Angular_Velocity_Buffer_Tag>;
using Center_Of_Mass_Buffer = CUDA_Array<float4, struct Center_Of_Mass_Buffer_Tag>;
using Predicted_Position_Buffer = CUDA_Array<float4, struct Predicted_Position_Buffer_Tag>;
using Predicted_Rotation_Buffer = CUDA_Array<float4, struct Predicted_Rotation_Buffer_Tag>;
using Bind_Pose_Position_Buffer = CUDA_Array<float4, struct Bind_Pose_Position_Buffer_Tag>;
using Bind_Pose_Inverse_Bind_Pose_Buffer = CUDA_Array<float4, struct Bind_Pose_Inverse_Bind_Pose_Buffer_Tag>;
using Bind_Pose_Center_Of_Mass_Buffer = CUDA_Array<float4, struct Bind_Pose_Center_Of_Mass_Buffer_Tag>;
using Mass_Buffer = CUDA_Array<float, struct Mass_Buffer_Tag>;
using Size_Buffer = CUDA_Array<float4, struct Size_Buffer_Tag>;
using Density_Buffer = CUDA_Array<float, struct Density_Buffer_Tag>;
using Particle_Correction_Info_Buffer = CUDA_Array<Particle_Correction_Info, struct Particle_Correction_Info_Buffer_Tag>;

using New_Position_Buffer = CUDA_Array<float4, struct New_Position_Buffer_Tag>;
using New_Rotation_Buffer = CUDA_Array<float4, struct New_Rotation_Buffer_Tag>;
using New_Goal_Position_Buffer = CUDA_Array<float4, struct New_Goal_Position_Buffer_Tag>;
