#!/usr/bin/env python3
"""
Cluster-Level Material Selection Pipeline

This script performs material selection at the cluster level instead of individual objects.
For each cluster (geometry + color), it:
1. Selects candidate objects (at least 3, or 1/3 of cluster size)
2. Generates material-painted previews for candidates
3. Uses VLM to pick the best material-painted result
4. Applies the winning material to all objects in the cluster

Usage:
    python cluster_material_selection.py --scene <scene_name> --cluster_file <cluster_json>
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

# Import VLM utilities
try:
    from utils.qwen_model_manager import ensure_model_loaded, QWEN_AVAILABLE
    from utils.qwen_query_materials import process_image_qwen_multi
except ImportError:
    # Fallback if imports fail
    QWEN_AVAILABLE = False
    def ensure_model_loaded():
        pass
    def process_image_qwen_multi(images, prompt, temperature=0.1):
        return None

def select_cluster_candidates(cluster, min_candidates=3, max_candidates=8):
    """
    Select candidate objects from a cluster based on the user's logic:
    - 1/3 of cluster size, but minimum 3 and maximum 8 candidates
    - If cluster has < 3 objects, use all of them
    - If cluster has many objects, use up to 8 candidates max

    Args:
        cluster: List of object names in the cluster
        min_candidates: Minimum number of candidates to select (default: 3)
        max_candidates: Maximum number of candidates to select (default: 8)

    Returns:
        list: Selected candidate object names
    """
    cluster_size = len(cluster)

    if cluster_size <= min_candidates:
        # Use all objects if cluster is small
        return cluster
    else:
        # Use 1/3 of cluster, but clamp between min_candidates and max_candidates
        num_candidates = max(min_candidates, min(max_candidates, cluster_size // 3))
        # Select evenly spaced candidates for diversity
        indices = np.linspace(0, cluster_size - 1, num_candidates, dtype=int)
        return [cluster[i] for i in indices]

def load_clustering_results(scene_name):
    """
    Load clustering results from the retrieval pipeline.
    Prefers geometry+color clustering for material painting optimization.

    Args:
        scene_name: Name of the scene

    Returns:
        dict: Clustering results with cluster assignments
    """
    # First, try to load geometry+color clustering (preferred for material painting)
    geom_color_cache_file = f"cache/clustering_cache/{scene_name}/clustering_results_geom_color.json"

    if os.path.exists(geom_color_cache_file):
        with open(geom_color_cache_file, 'r') as f:
            results = json.load(f)
        print(f"✅ Loaded geometry+color clustering results for material painting")
        return results

    # Fallback: try geometry-only clustering
    geom_only_cache_file = f"cache/clustering_cache/{scene_name}/clustering_results.json"

    if os.path.exists(geom_only_cache_file):
        with open(geom_only_cache_file, 'r') as f:
            results = json.load(f)
        print(f"⚠️  Using geometry-only clustering results (geometry+color not available)")
        return results

    # If no cache exists, generate geometry+color clustering
    print(f"🔄 Generating geometry+color clustering for material painting...")
    from identical_clustering import cluster_chairs
    chair_clusters = cluster_chairs(scene_name, use_color_clustering=True)

    if chair_clusters:
        # Convert to expected format
        results = {
            "scene_name": scene_name,
            "clusters": chair_clusters,
            "clustering_method": "geometry_color_hybrid",
            "total_objects": sum(len(c) for c in chair_clusters),
            "num_clusters": len(chair_clusters),
            "cluster_sizes": [len(c) for c in chair_clusters],
            "use_color_clustering": True
        }

        # Save to cache for future use
        os.makedirs(os.path.dirname(geom_color_cache_file), exist_ok=True)
        with open(geom_color_cache_file, 'w') as f:
            json.dump(results, f, indent=4)

        print(f"✅ Generated and saved geometry+color clustering results")
        return results

    return None

def get_material_painted_preview(obj_path, obj_name):
    """
    Get the path to the material-painted rendering preview for an object.

    Args:
        obj_path: Path to the object directory
        obj_name: Name of the object

    Returns:
        str: Path to the rendering preview, or None if not found
    """
    # Look for render_image_0.png in the material painting results
    preview_path = f"{obj_path}/A-ReTextured/OT_refined_with_adaptation/render_image_0.png"

    if os.path.exists(preview_path):
        return preview_path

    return None

def create_cluster_comparison_image(cluster_candidates, scene_name, cluster_id):
    """
    Create a comparison image showing all candidate material-painted results side by side.

    Args:
        cluster_candidates: List of (obj_name, preview_path) tuples
        scene_name: Name of the scene
        cluster_id: ID of the cluster

    Returns:
        str: Path to the comparison image
    """
    if not cluster_candidates:
        return None

    # Load all preview images
    images = []
    valid_candidates = []

    for obj_name, preview_path in cluster_candidates:
        if preview_path and os.path.exists(preview_path):
            try:
                img = Image.open(preview_path).convert('RGB')
                images.append(img)
                valid_candidates.append((obj_name, preview_path))
            except Exception as e:
                print(f"Warning: Failed to load preview for {obj_name}: {e}")
                continue

    if len(images) < 2:
        print(f"Warning: Need at least 2 valid previews for cluster {cluster_id}, got {len(images)}")
        return None

    # Resize all images to the same height (maintain aspect ratio)
    min_height = min(img.size[1] for img in images)
    resized_images = []

    for img in images:
        aspect_ratio = img.size[0] / img.size[1]
        new_width = int(min_height * aspect_ratio)
        resized_img = img.resize((new_width, min_height), Image.Resampling.LANCZOS)
        resized_images.append(resized_img)

    # Create side-by-side comparison
    total_width = sum(img.size[0] for img in resized_images)
    max_height = max(img.size[1] for img in resized_images)

    comparison_image = Image.new('RGB', (total_width, max_height), 'white')

    x_offset = 0
    for i, img in enumerate(resized_images):
        comparison_image.paste(img, (x_offset, 0))

        # Add label with candidate number
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(comparison_image)
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 40)
        except:
            font = ImageFont.load_default()

        label = f"#{i+1}"
        bbox = draw.textbbox((0, 0), label, font=font)
        label_width = bbox[2] - bbox[0]
        label_height = bbox[3] - bbox[1]

        # Draw label background
        draw.rectangle([x_offset + 10, 10, x_offset + 10 + label_width + 10, 10 + label_height + 10], fill='black')
        draw.text((x_offset + 15, 15), label, fill='white', font=font)

        x_offset += img.size[0]

    # Save comparison image
    output_dir = f"output/mat_painting_stage/{scene_name}/cluster_material_selection"
    os.makedirs(output_dir, exist_ok=True)
    comparison_path = f"{output_dir}/cluster_{cluster_id}_comparison.jpg"
    comparison_image.save(comparison_path, quality=90)

    return comparison_path

def select_best_material_with_vlm(comparison_image_path, cluster_candidates, stitched_images, cluster_id):
    """
    Use VLM to select the best material-painted result from the cluster candidates.

    Args:
        comparison_image_path: Path to the comparison image
        cluster_candidates: List of (obj_name, preview_path) tuples
        stitched_images: List of stitched image paths for context
        cluster_id: ID of the cluster

    Returns:
        int: Index of the winning candidate (0-based)
    """
    if not QWEN_AVAILABLE:
        print("Warning: Qwen model not available, falling back to random selection")
        return 0

    # Prepare images for VLM: stitched images + comparison image
    image_paths = stitched_images + [comparison_image_path]

    prompt = f"""You are evaluating different material painting results for a cluster of similar objects (cluster {cluster_id}).

I have prepared several material-painted versions of the same type of object. Each version shows how the material looks when applied to the object.

Looking at the comparison image, evaluate which material-painted version (labeled #1, #2, #3, etc.) best represents a realistic and appropriate material for objects of this type in a typical indoor environment.

Consider:
- How natural and realistic the material appears
- Whether it would fit well in a professional or home setting
- The overall quality and appropriateness of the material choice

Respond with ONLY the number of your chosen option (e.g., "2" or "3"). Do not include any other text or explanation."""

    try:
        ensure_model_loaded()
        response = process_image_qwen_multi(image_paths, prompt, temperature=0.1)

        if response and response.strip():
            # Extract the number from the response
            import re
            numbers = re.findall(r'\d+', response.strip())
            if numbers:
                choice = int(numbers[0]) - 1  # Convert to 0-based index
                if 0 <= choice < len(cluster_candidates):
                    return choice

        print(f"Warning: Invalid VLM response for cluster {cluster_id}: {response}")

    except Exception as e:
        print(f"Error during VLM evaluation for cluster {cluster_id}: {e}")

    # Fallback to first candidate
    return 0

def apply_winning_material_to_cluster(winning_candidate, cluster, scene_name, cluster_id):
    """
    Apply the winning material-painted result to all objects in the cluster.

    Args:
        winning_candidate: Tuple of (obj_name, preview_path) for the winner
        cluster: List of all object names in the cluster
        scene_name: Name of the scene
        cluster_id: ID of the cluster

    Returns:
        dict: Results of the material application
    """
    winning_obj_name, winning_preview_path = winning_candidate
    winning_obj_path = f"output/mat_painting_stage/{scene_name}/{winning_obj_name}"

    results = {
        "cluster_id": cluster_id,
        "winning_candidate": winning_obj_name,
        "applied_to": [],
        "success_count": 0,
        "failure_count": 0
    }

    print(f"  Applying winning material from {winning_obj_name} to cluster {cluster_id}")

    for obj_name in cluster:
        if obj_name == winning_obj_name:
            # Skip the winner itself
            results["applied_to"].append(f"{obj_name} (original)")
            results["success_count"] += 1
            continue

        target_path = f"output/mat_painting_stage/{scene_name}/{obj_name}"

        try:
            # Copy the winning material painting results to this object
            winning_material_dir = f"{winning_obj_path}/A-ReTextured/OT_refined_with_adaptation"
            target_material_dir = f"{target_path}/A-ReTextured/OT_refined_with_adaptation"

            if os.path.exists(winning_material_dir):
                # Create target directory
                os.makedirs(os.path.dirname(target_material_dir), exist_ok=True)

                # Copy the winning material results
                if os.path.exists(target_material_dir):
                    shutil.rmtree(target_material_dir)

                shutil.copytree(winning_material_dir, target_material_dir)

                results["applied_to"].append(obj_name)
                results["success_count"] += 1
                print(f"    Applied to {obj_name}")
            else:
                results["applied_to"].append(f"{obj_name} (failed - no source)")
                results["failure_count"] += 1
                print(f"    Failed to apply to {obj_name} (no source material)")

        except Exception as e:
            results["applied_to"].append(f"{obj_name} (error: {str(e)})")
            results["failure_count"] += 1
            print(f"    Error applying to {obj_name}: {e}")

    return results


def propagate_material_to_non_candidates(winning_obj_path, target_objects, scene_name):
    """
    For objects that didn't go through material painting, propagate the winning material.

    This function handles objects that were not selected as candidates in cluster-aware
    processing. It copies:
    1. The winning material's PBR textures (A-ReTextured directory)
    2. The gpt4_query results (material selections)

    Args:
        winning_obj_path: Full path to the winning object's directory
        target_objects: List of object names to propagate to (non-candidates)
        scene_name: Name of the scene

    Returns:
        dict: Results of the propagation with 'success' and 'failed' lists
    """
    results = {"success": [], "failed": []}

    winning_material_dir = os.path.join(winning_obj_path, "A-ReTextured")
    winning_gpt_dir = os.path.join(winning_obj_path, "gpt4_query")

    if not os.path.exists(winning_material_dir):
        print(f"    Warning: No A-ReTextured directory found in winner: {winning_obj_path}")
        return results

    parent_dir = os.path.dirname(winning_obj_path)

    for target_obj in target_objects:
        target_path = os.path.join(parent_dir, target_obj)

        try:
            # Ensure target object directory exists
            if not os.path.exists(target_path):
                # Try to prepare the object from object_stage
                object_stage_path = target_path.replace("mat_painting_stage", "object_stage")
                if os.path.exists(object_stage_path):
                    # Import and use prepare_for_painting
                    import subprocess
                    python_cmd = os.environ.get("LITEREALITY_PYTHON", sys.executable)
                    subprocess.run(
                        [python_cmd, "litereality/LR_retrieval/prepare_for_painting.py",
                         "--single-object", object_stage_path],
                        capture_output=True,
                        check=False
                    )

            if not os.path.exists(target_path):
                os.makedirs(target_path, exist_ok=True)

            # Copy A-ReTextured directory (complete material pipeline output)
            target_material_dir = os.path.join(target_path, "A-ReTextured")
            if os.path.exists(target_material_dir):
                shutil.rmtree(target_material_dir)
            shutil.copytree(winning_material_dir, target_material_dir)

            # Copy gpt4_query directory if it exists (material selection logs)
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


def run_object_showcase_for_propagated(target_path, image_size=500):
    """
    Run the Object_showcase.py script to generate render_image_0.png for a propagated object.

    This is needed for objects that received propagated materials but didn't go through
    the full material painting pipeline.

    Args:
        target_path: Path to the object directory
        image_size: Size of the rendered image (default: 500)

    Returns:
        bool: True if successful
    """
    try:
        import subprocess

        # Use Blender to run Object_showcase.py
        result = subprocess.run(
            ["blender", "-b", "-P", "litereality/LR_mat_painting/Object_showcase.py",
             "--", "--scene", target_path, "--image_size", str(image_size)],
            capture_output=True,
            text=True,
            check=False
        )

        # Check if render was created
        render_path = os.path.join(target_path, "A-ReTextured", "OT_refined_with_adaptation", "render_image_0.png")
        return os.path.exists(render_path)

    except Exception as e:
        print(f"    Error running Object_showcase for {target_path}: {e}")
        return False

def process_cluster_material_selection(scene_name, clustering_results=None):
    """
    Main function to perform cluster-level material selection.

    Args:
        scene_name: Name of the scene to process
        clustering_results: Optional pre-loaded clustering results

    Returns:
        dict: Summary of the cluster material selection process
    """
    print(f"\n🎨 Starting Cluster-Level Material Selection for scene: {scene_name}")

    # Load clustering results
    if clustering_results is None:
        clustering_results = load_clustering_results(scene_name)

    if not clustering_results or "clusters" not in clustering_results:
        print("❌ No clustering results found")
        return None

    clusters = clustering_results["clusters"]
    print(f"📊 Found {len(clusters)} clusters to process")

    # Collect stitched images for VLM context
    stitched_images = []
    for root, dirs, files in os.walk(f"output/object_stage/{scene_name}"):
        if "stitched_image.jpg" in files:
            stitched_images.append(os.path.join(root, "stitched_image.jpg"))

    print(f"📸 Found {len(stitched_images)} stitched images for context")

    overall_results = {
        "scene_name": scene_name,
        "total_clusters": len(clusters),
        "processed_clusters": 0,
        "cluster_results": []
    }

    for cluster_id, cluster in enumerate(clusters):
        print(f"\n🔄 Processing Cluster {cluster_id + 1}/{len(clusters)}: {len(cluster)} objects")

        if len(cluster) < 2:
            print(f"  ⏭️  Skipping cluster with only {len(cluster)} object(s)")
            continue

        # Select candidates for this cluster
        candidates = select_cluster_candidates(cluster)
        print(f"  🎯 Selected {len(candidates)} candidates: {candidates}")

        # Get material-painted previews for candidates
        cluster_candidates = []
        for obj_name in candidates:
            obj_path = f"output/mat_painting_stage/{scene_name}/{obj_name}"
            preview_path = get_material_painted_preview(obj_path, obj_name)

            if preview_path:
                cluster_candidates.append((obj_name, preview_path))
            else:
                print(f"  ⚠️  No material preview found for {obj_name}")

        if len(cluster_candidates) < 2:
            print(f"  ⏭️  Skipping cluster (only {len(cluster_candidates)} valid previews)")
            continue

        # Create comparison image
        comparison_path = create_cluster_comparison_image(cluster_candidates, scene_name, cluster_id)

        if not comparison_path:
            print(f"  ❌ Failed to create comparison image for cluster {cluster_id}")
            continue

        print(f"  🖼️  Created comparison image: {comparison_path}")

        # Use VLM to select the best material
        winning_idx = select_best_material_with_vlm(
            comparison_path,
            cluster_candidates,
            stitched_images[:3],  # Use up to 3 stitched images for context
            cluster_id
        )

        winning_candidate = cluster_candidates[winning_idx]
        print(f"  🏆 VLM selected candidate #{winning_idx + 1}: {winning_candidate[0]}")

        # Apply winning material to entire cluster
        application_results = apply_winning_material_to_cluster(
            winning_candidate, cluster, scene_name, cluster_id
        )

        cluster_result = {
            "cluster_id": cluster_id,
            "cluster_size": len(cluster),
            "candidates_selected": len(candidates),
            "candidates_with_previews": len(cluster_candidates),
            "winning_candidate": winning_candidate[0],
            "vlm_selection_index": winning_idx,
            "application_results": application_results
        }

        overall_results["cluster_results"].append(cluster_result)
        overall_results["processed_clusters"] += 1

        print(f"  📊 Applied to {application_results['success_count']}/{len(cluster)} objects")

    # Save overall results
    output_dir = f"output/mat_painting_stage/{scene_name}/cluster_material_selection"
    os.makedirs(output_dir, exist_ok=True)

    results_file = f"{output_dir}/cluster_material_selection_results.json"
    with open(results_file, 'w') as f:
        json.dump(overall_results, f, indent=4)

    print(f"\n✅ Cluster material selection complete!")
    print(f"📄 Results saved to: {results_file}")
    print(f"📊 Processed {overall_results['processed_clusters']}/{overall_results['total_clusters']} clusters")

    return overall_results

def main():
    parser = argparse.ArgumentParser(description="Cluster-Level Material Selection Pipeline")
    parser.add_argument("--scene", type=str, required=True, help="Scene name to process")
    parser.add_argument("--cluster_file", type=str, help="Optional path to clustering results JSON file")

    args = parser.parse_args()

    clustering_results = None
    if args.cluster_file and os.path.exists(args.cluster_file):
        with open(args.cluster_file, 'r') as f:
            clustering_results = json.load(f)

    process_cluster_material_selection(args.scene, clustering_results)

if __name__ == "__main__":
    main()