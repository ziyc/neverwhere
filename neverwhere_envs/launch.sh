#!/bin/bash

# Function to launch a process for a given GPU and scene
nwscene_maker() {
    local scene_dir=$1
    local scene_name=$2
    local downscale_factor=$3

    echo "RUNNING: Processing scene $scene_name from directory $scene_dir with downscale factor $downscale_factor"
    python neverwhere_envs/launch.py \
        --scene-name $scene_name \
        --dataset-dir $scene_dir \
        --gpu-index 0 \
        --frame-downsample 2 \
        --frame-downsample-threshold 400 \
        --image-downscale $downscale_factor \
        --geo-keys depth confidence \
        --gs-type 3dgs \
        --refine-mesh
}

# Export function so it can be used by other scripts
export nwscene_maker