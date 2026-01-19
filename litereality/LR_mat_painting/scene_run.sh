#!/bin/bash
set -e  # Exit immediately if any command fails

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -d|--dir) PARENT_DIR="$2"; shift ;;  # Accepts -d or --dir as flag
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Check if PARENT_DIR is provided
if [ -z "$PARENT_DIR" ]; then
    echo "Error: No directory specified. Use -d or --dir to provide the path."
    exit 1
fi

scene_name="$(basename "$PARENT_DIR")"

# Get the absolute path to the run.sh script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
RUN_SCRIPT="${SCRIPT_DIR}/run.sh"

# Get project root for Python commands
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." &> /dev/null && pwd )"
# Use Python from environment variable or default to 'python'
PYTHON_CMD="${LITEREALITY_PYTHON:-python}"

# Determine source directory (object_stage) - iterate over objects that need processing
# PARENT_DIR is mat_painting_stage, but we need to iterate over object_stage to find all objects
OBJECT_STAGE_DIR="$PARENT_DIR"
OBJECT_STAGE_DIR="${OBJECT_STAGE_DIR/mat_painting_stage/object_stage}"

# Check if object_stage exists, otherwise use mat_painting_stage (backward compatibility)
if [ ! -d "$OBJECT_STAGE_DIR" ]; then
    echo "Warning: object_stage directory not found: $OBJECT_STAGE_DIR"
    echo "   Using mat_painting_stage directory instead (objects may already be prepared)"
    OBJECT_STAGE_DIR="$PARENT_DIR"
fi

# Count objects and categorize them
total_folders=0
chair_count=0
non_chair_count=0
chair_list=""

for folder in "$OBJECT_STAGE_DIR"/*/; do
    [ -d "$folder" ] || continue
    folder_name=$(basename "$folder")
    total_folders=$((total_folders + 1))
    if [[ "${folder_name,,}" == chair* ]]; then
        chair_count=$((chair_count + 1))
        chair_list="$chair_list $folder_name"
    else
        non_chair_count=$((non_chair_count + 1))
    fi
done

# Log file to store execution times (save to output directory)
LOG_FILE="output/mat_painting_stage/${scene_name}/processing_times.log"

# Create the log directory if it doesn't exist
mkdir -p "$(dirname "$LOG_FILE")"

# Clear the log file before starting
echo "Processing Times Log - $(date)" > "$LOG_FILE"

# Initialize timing variables
chair_elapsed_time=0
total_non_chair_time=0

echo "======================================================"
echo "CLUSTER-AWARE MATERIAL PAINTING PIPELINE"
echo "Scene: $scene_name"
echo "Source: $OBJECT_STAGE_DIR"
echo "Target: $PARENT_DIR"
echo "======================================================"

# =====================================================================
# Phase 1: Process CHAIR objects with cluster-aware pipeline FIRST
# =====================================================================
echo ""
echo "======================================================"
echo "PHASE 1: Processing Chair objects with cluster-aware pipeline"
echo "======================================================"

if [ "$chair_count" -eq 0 ]; then
    echo "No Chair objects found. Skipping cluster-aware processing."
else
    echo "Found $chair_count Chair objects"
    echo "Objects: $chair_list"
    echo ""

    # Record start time for clustering phase
    chair_start_time=$(date +%s)

    # Step 1.1: Generate/load geometry+color clustering
    echo "Loading/generating geometry+color clustering..."
    cd "$PROJECT_ROOT"
    $PYTHON_CMD -c "
import sys
sys.path.insert(0, '.')
from litereality.LR_retrieval.identical_clustering import ensure_geometry_color_clustering
result = ensure_geometry_color_clustering('$scene_name')
if result:
    print(f'Clustering ready: {result[\"num_clusters\"]} clusters')
else:
    print('Warning: Could not generate clustering')
"

    # Step 1.2: Run cluster-aware processing
    echo ""
    echo "Running cluster-aware material painting..."
    $PYTHON_CMD litereality/LR_mat_painting/cluster_aware_processing.py \
        --scene "$scene_name" \
        --parent-dir "$PARENT_DIR"

    # Record end time for clustering phase
    chair_end_time=$(date +%s)
    chair_elapsed_time=$((chair_end_time - chair_start_time))

    # Format chair processing time
    chair_hours=$((chair_elapsed_time / 3600))
    chair_minutes=$(( (chair_elapsed_time % 3600) / 60 ))
    chair_seconds=$((chair_elapsed_time % 60))
    formatted_chair_time=""
    [[ $chair_hours -gt 0 ]] && formatted_chair_time="${chair_hours}h "
    [[ $chair_minutes -gt 0 ]] && formatted_chair_time="${formatted_chair_time}${chair_minutes}m "
    formatted_chair_time="${formatted_chair_time}${chair_seconds}s"

    echo "Chair clustering completed in: $formatted_chair_time"
fi

echo ""
echo "======================================================"
echo "Phase 1 Complete: Chair Objects (Clustering)"
echo "  - Total Chair objects: $chair_count"
echo "  - Time taken: $formatted_chair_time"
echo "======================================================"

# =====================================================================
# Phase 2: Process NON-CHAIR objects individually
# =====================================================================
echo ""
echo "======================================================"
echo "PHASE 2: Processing non-Chair objects individually"
echo "======================================================"

current_folder=0
total_non_chair_time=0
skipped_folders=0
processed_non_chairs=0

# Process non-Chair objects individually
for folder in "$OBJECT_STAGE_DIR"/*/; do
    folder_name=$(basename "$folder")

    # Skip Chair objects - they were processed in Phase 1
    if [[ "${folder_name,,}" == chair* ]]; then
        continue
    fi

    current_folder=$((current_folder + 1))

    # Display progress
    echo "----------------------------------------------------"
    echo "[$current_folder/$non_chair_count] Processing: $folder_name"

    # Check if this object has already been processed completely
    output_path="output/mat_painting_stage/${scene_name}/${folder_name}/A-ReTextured/OT_refined_with_adaptation"

    # Check specifically for render_image_0.png which indicates the process completed successfully
    if [ -f "${output_path}/render_image_0.png" ]; then
        echo "Skipping $folder_name - already processed (found render_image_0.png)"
        skipped_folders=$((skipped_folders + 1))
        echo "$folder_name: SKIPPED (already processed) - SUCCESS" >> "$LOG_FILE"
        continue
    fi

    # Additional check for other completion indicators
    if [ -d "${output_path}" ] && [ "$(ls -A "${output_path}" 2>/dev/null)" ]; then
        render_count=$(find "${output_path}" -name "render_image_*.png" 2>/dev/null | wc -l)
        if [ "$render_count" -gt 0 ]; then
            echo "Skipping $folder_name - already processed (found $render_count render images)"
            skipped_folders=$((skipped_folders + 1))
            echo "$folder_name: SKIPPED (found $render_count render images) - SUCCESS" >> "$LOG_FILE"
            continue
        fi
    fi

    # Skip windows and doors if parent wall was processed
    if [[ "$folder_name" =~ ^Wall[0-9]+_(Window|Door)_[0-9]+$ ]]; then
        wall_name=$(echo "$folder_name" | sed -E 's/^(Wall[0-9]+)_.*/\1/')
        wall_output_path="output/mat_painting_stage/${scene_name}/${wall_name}/A-ReTextured/OT_refined_with_adaptation"

        if [ -f "${wall_output_path}/render_image_0.png" ]; then
            echo "Skipping $folder_name - parent wall ($wall_name) already processed"
            skipped_folders=$((skipped_folders + 1))
            echo "$folder_name: SKIPPED (parent wall $wall_name already processed) - SUCCESS" >> "$LOG_FILE"
            continue
        fi
    fi

    # Record start time
    start_time=$(date +%s)

    # Determine the prepared object path in mat_painting_stage
    PREPARED_OBJECT_PATH="$PARENT_DIR/$folder_name"

    # Prepare this object for material painting
    echo "Preparing object for material painting..."

    OBJECT_SOURCE_PATH="$folder"

    if [ ! -d "$OBJECT_SOURCE_PATH" ]; then
        echo "Warning: Source object not found: $OBJECT_SOURCE_PATH"
        if [ -d "$PREPARED_OBJECT_PATH" ]; then
            echo "Object already exists in mat_painting_stage, continuing"
        else
            echo "Skipping this object..."
            continue
        fi
    else
        cd "$PROJECT_ROOT"
        if $PYTHON_CMD litereality/LR_retrieval/prepare_for_painting.py --single-object "$OBJECT_SOURCE_PATH"; then
            echo "Object prepared successfully"
        else
            if [ -d "$PREPARED_OBJECT_PATH" ]; then
                echo "Object exists, continuing with material painting"
            else
                echo "Skipping material painting for this object..."
                continue
            fi
        fi
    fi

    # Verify the object exists before processing
    if [ ! -d "$PREPARED_OBJECT_PATH" ]; then
        echo "Warning: Prepared object not found at $PREPARED_OBJECT_PATH"
        continue
    fi

    # Run the material painting script
    if bash "$RUN_SCRIPT" "$PREPARED_OBJECT_PATH"; then
        echo "Successfully processed: $folder_name"
    else
        echo "Error processing: $folder_name. Moving to the next one."
    fi

    # Record end time
    end_time=$(date +%s)
    elapsed_time=$((end_time - start_time))
    total_non_chair_time=$((total_non_chair_time + elapsed_time))

    # Format time
    hours=$((elapsed_time / 3600))
    minutes=$(( (elapsed_time % 3600) / 60 ))
    seconds=$((elapsed_time % 60))
    formatted_time=""
    [[ $hours -gt 0 ]] && formatted_time="${hours}h "
    [[ $minutes -gt 0 ]] && formatted_time="${formatted_time}${minutes}m "
    formatted_time="${formatted_time}${seconds}s"

    # Check if processing was successful
    success_check_path="$PREPARED_OBJECT_PATH/A-ReTextured/OT_refined_with_adaptation/render_image_0.png"
    if [ -f "$success_check_path" ]; then
        status="SUCCESS"
        processed_non_chairs=$((processed_non_chairs + 1))
    else
        status="FAILED"
    fi

    echo "Time taken: $formatted_time ($status)"
    echo "$folder_name: $elapsed_time seconds (${formatted_time}) - $status" >> "$LOG_FILE"

    echo "----------------------------------------------------"
done

echo ""
echo "======================================================"
echo "Phase 2 Complete: Non-Chair Objects"
echo "  - Total non-Chair objects: $non_chair_count"
echo "  - Processed: $processed_non_chairs"
echo "  - Skipped (already done): $skipped_folders"
echo "======================================================"

# =====================================================================
# Summary
# =====================================================================

# Format non-chair processing time
non_chair_hours=$((total_non_chair_time / 3600))
non_chair_minutes=$(( (total_non_chair_time % 3600) / 60 ))
non_chair_seconds=$((total_non_chair_time % 60))
formatted_non_chair_time=""
[[ $non_chair_hours -gt 0 ]] && formatted_non_chair_time="${non_chair_hours}h "
[[ $non_chair_minutes -gt 0 ]] && formatted_non_chair_time="${formatted_non_chair_time}${non_chair_minutes}m "
formatted_non_chair_time="${formatted_non_chair_time}${non_chair_seconds}s"

# Calculate total time (sum of both phases)
total_time_taken=$((chair_elapsed_time + total_non_chair_time))
t_hours=$((total_time_taken / 3600))
t_minutes=$(( (total_time_taken % 3600) / 60 ))
t_seconds=$((total_time_taken % 60))
formatted_total=""
[[ $t_hours -gt 0 ]] && formatted_total="${t_hours}h "
[[ $t_minutes -gt 0 ]] && formatted_total="${formatted_total}${t_minutes}m "
formatted_total="${formatted_total}${t_seconds}s"

echo ""
echo "======================================================"
echo "CLUSTER-AWARE MATERIAL PAINTING COMPLETE!"
echo "======================================================"
echo "Summary:"
echo "  - Total objects: $total_folders"
echo "  - Chair objects processed with clustering: $chair_count"
echo "  - Non-Chair objects processed individually: $non_chair_count"
echo "  - Time for chair clustering: ${formatted_chair_time:-0s}"
echo "  - Time for non-chair processing: ${formatted_non_chair_time:-0s}"
echo "  - Total processing time: $formatted_total"
echo "  - Detailed log saved to: $LOG_FILE"
echo "======================================================"
