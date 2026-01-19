#!/usr/bin/env python3
"""
Cluster-Aware Material Painting Pipeline

This module orchestrates efficient material painting by processing only cluster candidates
instead of all objects. For each cluster:
1. Select 3-8 candidate objects
2. Run full material painting only on candidates
3. Use VLM to pick the best candidate
4. Propagate winning material to all cluster members (skipping full processing for non-candidates)

Usage:
    python cluster_aware_processing.py --scene <scene_name> --parent-dir <mat_painting_stage_path>
"""

import os
import sys
import json
import shutil
import argparse
import subprocess
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from litereality.LR_retrieval.identical_clustering import (
    load_clustering_for_material_painting,
    ensure_geometry_color_clustering
)


def select_cluster_candidates(cluster, min_candidates=3, max_candidates=8):
    """
    Select candidate objects from a cluster.
    - 1/3 of cluster size, but minimum 3 and maximum 8 candidates
    - If cluster has < 3 objects, use all of them

    Args:
        cluster: List of object names in the cluster
        min_candidates: Minimum number of candidates (default: 3)
        max_candidates: Maximum number of candidates (default: 8)

    Returns:
        list: Selected candidate object names
    """
    import numpy as np

    cluster_size = len(cluster)

    if cluster_size <= min_candidates:
        return list(cluster)

    # Use 1/3 of cluster, clamped between min and max
    num_candidates = max(min_candidates, min(max_candidates, cluster_size // 3))

    # Select evenly spaced candidates for diversity
    indices = np.linspace(0, cluster_size - 1, num_candidates, dtype=int)
    return [cluster[i] for i in indices]


def is_object_processed(obj_path):
    """
    Check if an object has already been fully processed.

    Args:
        obj_path: Path to the object directory in mat_painting_stage

    Returns:
        bool: True if the object has render_image_0.png
    """
    render_path = os.path.join(obj_path, "A-ReTextured", "OT_refined_with_adaptation", "render_image_0.png")
    return os.path.exists(render_path)


def run_material_painting_single_object(obj_name, scene_name, parent_dir):
    """
    Run the full material painting pipeline on a single object.

    Args:
        obj_name: Name of the object (e.g., "Chair0")
        scene_name: Name of the scene
        parent_dir: Path to mat_painting_stage directory

    Returns:
        bool: True if successful
    """
    obj_path = os.path.join(parent_dir, obj_name)

    # Check if already processed
    if is_object_processed(obj_path):
        print(f"    Skipping {obj_name} - already processed")
        return True

    # Get script directory
    script_dir = Path(__file__).parent
    run_script = script_dir / "run.sh"

    print(f"    Running material painting for {obj_name}...")

    try:
        result = subprocess.run(
            ["bash", str(run_script), obj_path],
            cwd=str(PROJECT_ROOT),
            capture_output=False,
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"    Error processing {obj_name}: {e}")
        return False


def prepare_single_object(obj_name, scene_name):
    """
    Prepare a single object for material painting (copy from object_stage).

    Args:
        obj_name: Name of the object
        scene_name: Name of the scene

    Returns:
        str: Path to prepared object in mat_painting_stage, or None if failed
    """
    object_stage_path = f"output/object_stage/{scene_name}/{obj_name}"
    mat_painting_path = f"output/mat_painting_stage/{scene_name}/{obj_name}"

    # Check if already prepared
    if os.path.exists(mat_painting_path) and os.listdir(mat_painting_path):
        return mat_painting_path

    if not os.path.exists(object_stage_path):
        print(f"    Warning: Object not found in object_stage: {object_stage_path}")
        return None

    # Use prepare_for_painting.py to prepare the object
    python_cmd = os.environ.get("LITEREALITY_PYTHON", sys.executable)

    try:
        result = subprocess.run(
            [python_cmd, "litereality/LR_retrieval/prepare_for_painting.py",
             "--single-object", object_stage_path],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True
        )
        return mat_painting_path
    except subprocess.CalledProcessError as e:
        print(f"    Error preparing {obj_name}: {e}")
        return None


def propagate_material_to_non_candidates(winning_obj_path, target_objects, scene_name, parent_dir):
    """
    For objects that didn't go through material painting, propagate the winning material.

    This copies:
    1. The winning material's PBR textures (A-ReTextured directory)
    2. The gpt4_query results (material selections)

    Args:
        winning_obj_path: Path to the winning object's directory
        target_objects: List of object names to propagate to
        scene_name: Name of the scene
        parent_dir: Path to mat_painting_stage directory

    Returns:
        dict: Results of the propagation
    """
    # Add the script directory to path for local imports
    script_dir = Path(__file__).parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    # Try to use the function from cluster_material_selection if available
    try:
        from cluster_material_selection import propagate_material_to_non_candidates as cms_propagate
        return cms_propagate(winning_obj_path, target_objects, scene_name)
    except ImportError:
        pass

    # Fallback: inline implementation
    results = {"success": [], "failed": []}

    winning_material_dir = os.path.join(winning_obj_path, "A-ReTextured")
    winning_gpt_dir = os.path.join(winning_obj_path, "gpt4_query")

    if not os.path.exists(winning_material_dir):
        print(f"    Warning: No A-ReTextured directory found in winner")
        return results

    for target_obj in target_objects:
        target_path = os.path.join(parent_dir, target_obj)

        try:
            # Ensure target object is prepared (has basic structure)
            if not os.path.exists(target_path):
                prepared_path = prepare_single_object(target_obj, scene_name)
                if not prepared_path:
                    results["failed"].append(target_obj)
                    continue

            # Copy A-ReTextured directory
            target_material_dir = os.path.join(target_path, "A-ReTextured")
            if os.path.exists(target_material_dir):
                shutil.rmtree(target_material_dir)
            shutil.copytree(winning_material_dir, target_material_dir)

            # Copy gpt4_query directory if it exists
            if os.path.exists(winning_gpt_dir):
                target_gpt_dir = os.path.join(target_path, "gpt4_query")
                if os.path.exists(target_gpt_dir):
                    shutil.rmtree(target_gpt_dir)
                shutil.copytree(winning_gpt_dir, target_gpt_dir)

            results["success"].append(target_obj)
            print(f"    Propagated material to {target_obj}")

        except Exception as e:
            print(f"    Error propagating to {target_obj}: {e}")
            results["failed"].append(target_obj)

    return results


def process_single_cluster(cluster_id, cluster_members, scene_name, parent_dir, run_script_path):
    """
    Process a single cluster efficiently.

    Args:
        cluster_id: ID of the cluster
        cluster_members: List of object names in the cluster
        scene_name: Name of the scene
        parent_dir: Path to mat_painting_stage directory
        run_script_path: Path to run.sh script

    Returns:
        dict: Results of processing this cluster
    """
    print(f"\n{'='*60}")
    print(f"Processing Cluster {cluster_id}: {len(cluster_members)} objects")
    print(f"  Members: {cluster_members}")
    print(f"{'='*60}")

    result = {
        "cluster_id": cluster_id,
        "cluster_size": len(cluster_members),
        "candidates": [],
        "non_candidates": [],
        "winning_candidate": None,
        "status": "pending"
    }

    # Handle single-object clusters
    if len(cluster_members) <= 1:
        print(f"  Single-object cluster, processing normally...")
        if cluster_members:
            obj_name = cluster_members[0]
            # Prepare and process the single object
            prepare_single_object(obj_name, scene_name)
            run_material_painting_single_object(obj_name, scene_name, parent_dir)
            result["candidates"] = [obj_name]
            result["winning_candidate"] = obj_name
        result["status"] = "completed"
        return result

    # Select candidates
    candidates = select_cluster_candidates(cluster_members)
    non_candidates = [m for m in cluster_members if m not in candidates]

    result["candidates"] = candidates
    result["non_candidates"] = non_candidates

    print(f"  Selected {len(candidates)} candidates: {candidates}")
    print(f"  Non-candidates (will receive propagated material): {non_candidates}")

    # Step 1: Prepare all candidates
    print(f"\n  [Step 1] Preparing candidate objects...")
    for candidate in candidates:
        prepare_single_object(candidate, scene_name)

    # Step 2: Run material painting ONLY on candidates
    print(f"\n  [Step 2] Running material painting on candidates...")
    processed_candidates = []
    for candidate in candidates:
        success = run_material_painting_single_object(candidate, scene_name, parent_dir)
        if success:
            obj_path = os.path.join(parent_dir, candidate)
            if is_object_processed(obj_path):
                processed_candidates.append(candidate)

    if not processed_candidates:
        print(f"  Warning: No candidates were successfully processed!")
        result["status"] = "failed"
        return result

    print(f"  Successfully processed: {processed_candidates}")

    # Step 3: Use VLM to select the best candidate
    print(f"\n  [Step 3] Selecting best candidate with VLM...")
    winning_candidate = select_best_candidate_with_vlm(
        processed_candidates, scene_name, parent_dir, cluster_id
    )

    result["winning_candidate"] = winning_candidate
    print(f"  Winning candidate: {winning_candidate}")

    # Step 4: Propagate winning material to non-candidates
    if non_candidates:
        print(f"\n  [Step 4] Propagating winning material to non-candidates...")
        winning_obj_path = os.path.join(parent_dir, winning_candidate)
        propagation_results = propagate_material_to_non_candidates(
            winning_obj_path, non_candidates, scene_name, parent_dir
        )
        print(f"  Propagation results: {len(propagation_results['success'])} success, "
              f"{len(propagation_results['failed'])} failed")

    result["status"] = "completed"
    return result


def select_best_candidate_with_vlm(candidates, scene_name, parent_dir, cluster_id):
    """
    Use VLM to select the best material-painted candidate.

    Args:
        candidates: List of candidate object names that have been processed
        scene_name: Name of the scene
        parent_dir: Path to mat_painting_stage directory
        cluster_id: ID of the cluster

    Returns:
        str: Name of the winning candidate
    """
    if len(candidates) == 1:
        return candidates[0]

    # Add the script directory to path for local imports
    script_dir = Path(__file__).parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    try:
        # Import VLM utilities
        from utils.qwen_model_manager import ensure_model_loaded, QWEN_AVAILABLE
        from utils.qwen_query_materials import process_image_qwen_multi

        if not QWEN_AVAILABLE:
            print("    VLM not available, using first candidate")
            return candidates[0]

        # Collect preview images
        preview_images = []
        valid_candidates = []

        for candidate in candidates:
            preview_path = os.path.join(
                parent_dir, candidate,
                "A-ReTextured", "OT_refined_with_adaptation", "render_image_0.png"
            )
            if os.path.exists(preview_path):
                preview_images.append(preview_path)
                valid_candidates.append(candidate)

        if len(valid_candidates) < 2:
            return valid_candidates[0] if valid_candidates else candidates[0]

        # Create comparison image using local import
        try:
            from cluster_material_selection import create_cluster_comparison_image
        except ImportError:
            # Fallback: create inline comparison
            print("    Could not import cluster_material_selection, using first candidate")
            return valid_candidates[0]

        cluster_candidates = [(name, os.path.join(
            parent_dir, name, "A-ReTextured", "OT_refined_with_adaptation", "render_image_0.png"
        )) for name in valid_candidates]

        comparison_path = create_cluster_comparison_image(
            cluster_candidates, scene_name, cluster_id
        )

        if not comparison_path:
            return valid_candidates[0]

        # Query VLM
        prompt = f"""You are evaluating different material painting results for a cluster of similar furniture objects.

Looking at the comparison image, evaluate which version (labeled #1, #2, #3, etc.) has the best:
- Realistic and natural material appearance
- Appropriate texture and color for the object type
- Overall quality and consistency

Respond with ONLY the number of your chosen option (e.g., "2" or "3"). No explanation needed."""

        ensure_model_loaded()
        response = process_image_qwen_multi([comparison_path], prompt, temperature=0.1)

        if response:
            import re
            numbers = re.findall(r'\d+', response.strip())
            if numbers:
                choice = int(numbers[0]) - 1  # Convert to 0-based index
                if 0 <= choice < len(valid_candidates):
                    return valid_candidates[choice]

    except Exception as e:
        print(f"    VLM selection error: {e}")

    # Fallback: return first candidate
    return candidates[0]


def process_scene_clustered(scene_name, parent_dir):
    """
    Main entry point for cluster-aware material painting.

    Args:
        scene_name: Name of the scene
        parent_dir: Path to mat_painting_stage directory

    Returns:
        dict: Overall processing results
    """
    print(f"\n{'#'*70}")
    print(f"#  CLUSTER-AWARE MATERIAL PAINTING PIPELINE")
    print(f"#  Scene: {scene_name}")
    print(f"{'#'*70}\n")

    # Step 1: Load/generate geometry+color clustering
    print("Loading geometry+color clustering...")
    clustering = load_clustering_for_material_painting(scene_name)

    if not clustering or "clusters" not in clustering:
        print("No clustering results found. Falling back to individual processing.")
        return None

    clusters = clustering["clusters"]
    print(f"Found {len(clusters)} clusters to process")

    # Get script path
    script_dir = Path(__file__).parent
    run_script = script_dir / "run.sh"

    # Step 2: Process each cluster
    results = {
        "scene_name": scene_name,
        "total_clusters": len(clusters),
        "cluster_results": [],
        "total_objects_processed": 0,
        "total_objects_propagated": 0
    }

    for cluster_id, cluster_members in enumerate(clusters):
        cluster_result = process_single_cluster(
            cluster_id, cluster_members, scene_name, parent_dir, run_script
        )
        results["cluster_results"].append(cluster_result)

        # Update statistics
        results["total_objects_processed"] += len(cluster_result["candidates"])
        results["total_objects_propagated"] += len(cluster_result.get("non_candidates", []))

    # Save results
    results_file = os.path.join(parent_dir, "cluster_aware_processing_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\n{'#'*70}")
    print(f"#  CLUSTER-AWARE PROCESSING COMPLETE")
    print(f"#  Total clusters: {results['total_clusters']}")
    print(f"#  Objects fully processed: {results['total_objects_processed']}")
    print(f"#  Objects with propagated materials: {results['total_objects_propagated']}")
    print(f"#  Results saved to: {results_file}")
    print(f"{'#'*70}\n")

    return results


def get_chair_objects(scene_name, parent_dir):
    """
    Get list of Chair objects from the scene.

    Args:
        scene_name: Name of the scene
        parent_dir: Path to mat_painting_stage or object_stage directory

    Returns:
        list: List of Chair object names
    """
    # Try object_stage first
    object_stage = parent_dir.replace("mat_painting_stage", "object_stage")

    search_dir = object_stage if os.path.exists(object_stage) else parent_dir

    chairs = []
    if os.path.exists(search_dir):
        for item in os.listdir(search_dir):
            if item.lower().startswith("chair") and os.path.isdir(os.path.join(search_dir, item)):
                chairs.append(item)

    return sorted(chairs)


def get_non_chair_objects(scene_name, parent_dir):
    """
    Get list of non-Chair objects from the scene.

    Args:
        scene_name: Name of the scene
        parent_dir: Path to mat_painting_stage or object_stage directory

    Returns:
        list: List of non-Chair object names
    """
    object_stage = parent_dir.replace("mat_painting_stage", "object_stage")
    search_dir = object_stage if os.path.exists(object_stage) else parent_dir

    non_chairs = []
    if os.path.exists(search_dir):
        for item in os.listdir(search_dir):
            if os.path.isdir(os.path.join(search_dir, item)):
                if not item.lower().startswith("chair"):
                    non_chairs.append(item)

    return sorted(non_chairs)


def main():
    parser = argparse.ArgumentParser(description="Cluster-Aware Material Painting Pipeline")
    parser.add_argument("--scene", type=str, required=True, help="Scene name")
    parser.add_argument("--parent-dir", type=str, required=True,
                        help="Path to mat_painting_stage directory")
    parser.add_argument("--chairs-only", action="store_true",
                        help="Only process Chair objects with cluster-aware pipeline")

    args = parser.parse_args()

    # Process the scene
    process_scene_clustered(args.scene, args.parent_dir)


if __name__ == "__main__":
    main()
