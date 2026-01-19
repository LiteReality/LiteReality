#!/bin/bash
# Add path

set -e  # Stop the script if any command fails

# Get the script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." &> /dev/null && pwd )"

# Change to project root directory
cd "$PROJECT_ROOT"

# Accept the folder path as an argument
Folder="${1:-All_test/chair_pair_4}"  # Use default if no argument provided

# Set the Blender path
export PATH=$PATH:third_party/blender_dir/blender-3.6.0-linux-x64
# Add the project root to PYTHONPATH to make third_party imports work
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

# Use Python from environment variable or default to 'python'
PYTHON_CMD="${LITEREALITY_PYTHON:-python}"

# Get object name for display
object_name=$(basename "$Folder")
start_time=$(date +%s)

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║ 🎨 LiteReality Material Painting Pipeline                    ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Object: $object_name"
echo ""

echo "[Step 1/5] 🧩 Stitching images together (~ 30 seconds)"
echo "--------------------------------------------------------"
$PYTHON_CMD litereality/LR_mat_painting/Onboarding_stitich_image.py --scene "$Folder"

echo ""
echo "[Step 2/5] 🤖 VLM Material Selection Pipeline (~ 5-10 minutes)"
echo "--------------------------------------------------------"
echo "  Phase 1: Top 10 material selection (or fewer if less available)"
echo "  Phase 2: Pattern analysis → Top 4"
echo "  Phase 3: Color adaptation"
echo "  Phase 4: Final selection"
$PYTHON_CMD litereality/LR_mat_painting/Retrieval_material_with_LLM.py --scene "$Folder"

echo ""
echo "[Step 3/6] 🎨 Applying color refinements (~ 30 seconds)"
echo "--------------------------------------------------------"
$PYTHON_CMD litereality/LR_mat_painting/Material_refinements.py --scene "$Folder"

echo ""
echo "[Step 4/6] 📏 Resizing textures to 1000px (~ 30 seconds)"
echo "--------------------------------------------------------"
$PYTHON_CMD litereality/LR_mat_painting/resize_texture.py --folder "$Folder" --mat_size 1000 --single-object

echo ""
echo "[Step 5/6] 🖌️ Generating object showcase with Blender (~ 2-4 minutes)"
echo "--------------------------------------------------------"
blender -b -P litereality/LR_mat_painting/Object_showcase.py -- --scene "$Folder" --image_size 500

echo ""
echo "[Step 6/6] 📊 Collecting visualization data..."
echo "--------------------------------------------------------"
$PYTHON_CMD litereality/LR_mat_painting/collect_visualization_log.py --scene "$Folder"

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║ ✓ Material Painting Pipeline completed successfully!         ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo "Total time: ${elapsed_time}s"
echo "Results saved to: $Folder/"
echo ""
