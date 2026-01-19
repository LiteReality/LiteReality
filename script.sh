#!/bin/bash

set -e  # Exit immediately if any command fails
export PATH=$PATH:third_party/blender_dir/blender-3.6.0-linux-x64
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# Accept command-line arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <scene_raw_data> <name_of_the_scene>"
    echo "Example: $0 scans/2025_03_16_10_29_01 scene_2"
    exit 1
fi

scene_raw_data="$1"
name_of_the_scene="$2"

# Python interpreter path - uses current conda environment's python
PYTHON_CMD="python"

# Step 1: preprocess the raw data (includes camera data preparation for retrieval)
$PYTHON_CMD litereality/LR_preprocessing/preprocessing.py --raw "$scene_raw_data" --name "$name_of_the_scene"
$PYTHON_CMD litereality/LR_preprocessing/scene_parsing/scene_parsing.py --name "$name_of_the_scene"
$PYTHON_CMD litereality/LR_preprocessing/bbox_polish.py --scene  "$name_of_the_scene"

# Step 2: Run the retrieval
$PYTHON_CMD litereality/LR_retrieval/main_qwen.py --name "input/object_stage/$name_of_the_scene"

# Step 3: Materail Painting
bash litereality/LR_mat_painting/scene_run.sh -d "output/mat_painting_stage/$name_of_the_scene"
# Note: Resize now happens per-object during scene_run.sh (more efficient)
blender -b -P litereality/LR_mat_painting/apply_mat_rotate.py -- --path "output/mat_painting_stage/$name_of_the_scene" --image_size 1000

# Step 4: Procedural Reconstruction (Using Cycles for better quality) # can be changed to "BLENDER_EEVEE" to speed up
blender -b --python litereality/LR_procedural_recon/integration_blender_upgrade.py -- --scene $name_of_the_scene --render_engine CYCLES
# generate videos
python litereality/LR_procedural_recon/export_videos.py --name $name_of_the_scene --fps 5