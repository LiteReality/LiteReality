"""
Material Refinement System

This script provides color refinement for material textures by:
1. Extracting colors from reference images
2. Applying color transfer using optimal transport
3. Adjusting texture colors based on specified RGB values
"""

import numpy as np
import cv2
import os
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from glob import glob
import warnings
import json
import shutil
import argparse
import time
import re
from litereality.LR_mat_painting.utils.output_formatter import formatter

# Suppress the KMeans warning
warnings.filterwarnings("ignore", category=UserWarning)


def get_image_colors(image, num_colors=10):
    """
    Extract the main colors from an image using KMeans clustering.
    
    Args:
        image: The input image in BGR format
        num_colors: Number of color clusters to identify
        
    Returns:
        Array of color centroids
    """
    # Resize for faster processing
    image = cv2.resize(image, (100, 100))
    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=num_colors, random_state=0).fit(pixels)
    # Suppress verbose output
    return kmeans.cluster_centers_


def average_color_distribution(reference_folder, num_colors=10):
    """
    Calculate the average color distribution across multiple reference images.
    
    Args:
        reference_folder: Folder containing reference images
        num_colors: Number of color clusters to identify
        
    Returns:
        Array of combined color centroids
    """
    # Suppress verbose output
    color_list = []
    image_count = 0
    
    for img_path in glob(os.path.join(reference_folder, "*.jpg")):
        image = cv2.imread(img_path)
        if image is not None:
            # Suppress verbose output
            image_colors = get_image_colors(image, num_colors)
            color_list.append(image_colors)
            image_count += 1
    
    if not color_list:
        return np.array([[128, 128, 128]] * num_colors)  # Default gray colors
    
    # Suppress verbose output
    
    # Combine all colors and find the main clusters
    all_colors = np.vstack(color_list)
    kmeans = KMeans(n_clusters=num_colors, random_state=0).fit(all_colors)
    combined_colors = kmeans.cluster_centers_
    
    return combined_colors


def apply_color_transfer(source_colors, target_image):
    """
    Apply color transfer from source colors to target image using optimal transport.
    
    Args:
        source_colors: Array of source color centroids
        target_image: Target image to recolor
        
    Returns:
        Color-transferred image
    """
    # Suppress verbose output
    
    # Extract colors from target image
    target_pixels = target_image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=len(source_colors), random_state=0).fit(target_pixels)
    target_colors = kmeans.cluster_centers_
    # Suppress verbose output

    # Compute optimal color mapping
    cost_matrix = np.linalg.norm(target_colors[:, np.newaxis] - source_colors, axis=2)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    # Suppress verbose output

    # Apply the color mapping
    label_to_color = {i: source_colors[j] for i, j in zip(row_ind, col_ind)}
    new_pixels = np.array([label_to_color[label] for label in kmeans.predict(target_pixels)])
    result = new_pixels.reshape(target_image.shape)
    
    # Suppress verbose output
    return result


def apply_color_to_texture(source_image, rgb_color):
    """
    Apply a specific RGB color to a texture by shifting in LAB color space.
    
    Args:
        source_image: Source texture image
        rgb_color: Target RGB color to apply
        
    Returns:
        Color-adjusted texture
    """
    # Suppress verbose output
    
    # Convert source image to LAB color space
    source_image = source_image.astype(np.uint8)
    source_lab = cv2.cvtColor(source_image, cv2.COLOR_BGR2LAB)
    
    # Convert target RGB color to LAB
    target_lab = cv2.cvtColor(np.uint8([[rgb_color]]), cv2.COLOR_RGB2LAB)[0][0]
    
    # Split LAB channels and adjust them
    l, a, b = cv2.split(source_lab)
    l_shifted = cv2.add(l, target_lab[0] - np.mean(l))
    a_shifted = cv2.add(a, target_lab[1] - np.mean(a))
    b_shifted = cv2.add(b, target_lab[2] - np.mean(b))
    
    # Merge channels and convert back to BGR
    result_lab = cv2.merge([l_shifted, a_shifted, b_shifted])
    final_image = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
    
    # Suppress verbose output
    return final_image


def load_material_albedo(material_id):
    """
    Load albedo/basecolor map for a material.
    
    Args:
        material_id: Material ID (e.g., "Wood_WoodPlanks017")
    
    Returns:
        str: Path to albedo map, or None if not found
    """
    # Parse material ID to get category and name
    parts = material_id.split("_")
    if len(parts) < 2:
        return None
    
    category = parts[0]
    subcategory_and_name = "_".join(parts[1:])
    
    # Extract base name (remove trailing letters like "A", "B")
    base_name = subcategory_and_name.rstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    
    # Try different path patterns
    base_path = "litereality_database/PBR_materials/material_lib/pbr_maps/train"
    
    # Helper function to convert to ACG naming pattern
    def convert_to_acg_name(name):
        """Convert material name to ACG naming pattern (e.g., PaintedMetal001 -> acg_painted_metal_001)"""
        # Step 1: Convert camelCase to snake_case (e.g., PaintedMetal -> Painted_Metal)
        result = re.sub(r'(?<!^)(?=[A-Z])', '_', name)
        # Step 2: Lowercase everything
        result = result.lower()
        # Step 3: Handle special cases like paintedmetal -> painted_metal (in case camelCase conversion didn't work)
        result = result.replace('paintedmetal', 'painted_metal')
        # Step 4: Add underscore before numbers (e.g., painted_metal004 -> painted_metal_004)
        # Match one or more letters followed by one or more digits
        result = re.sub(r'([a-z]+)([0-9]+)', r'\1_\2', result)
        return f'acg_{result}'
    
    # Generate ACG names for both full name and base name
    acg_name = convert_to_acg_name(subcategory_and_name)
    acg_base_name = convert_to_acg_name(base_name) if base_name != subcategory_and_name else None
    
    possible_paths = [
        f"{base_path}/{category}/{base_name}/basecolor.png",
        f"{base_path}/{category}/{base_name}/diffuse.png",
        f"{base_path}/{category}/{subcategory_and_name}/basecolor.png",
        f"{base_path}/{category}/{subcategory_and_name}/diffuse.png",
        f"{base_path}/{category}/{material_id}/basecolor.png",
        f"{base_path}/{category}/{material_id}/diffuse.png",
        f"{base_path}/{category}/{acg_name}/basecolor.png",
        f"{base_path}/{category}/{acg_name}/diffuse.png",
    ]
    
    # Add ACG base name paths if different from full name
    if acg_base_name and acg_base_name != acg_name:
        possible_paths.extend([
            f"{base_path}/{category}/{acg_base_name}/basecolor.png",
            f"{base_path}/{category}/{acg_base_name}/diffuse.png",
        ])
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None


def color_adapt_multiple_materials(scene_path, part_id, top_4_materials, target_rgb):
    """
    Phase 3: Apply color adaptation to all 4 materials.
    
    Args:
        scene_path: Path to scene folder
        part_id: Part identifier (e.g., "solid_001")
        top_4_materials: List of 4 material IDs
        target_rgb: Target RGB color tuple (R, G, B)
    
    Returns:
        dict: Color adaptation results with paths to adapted materials
    """
    from litereality.LR_mat_painting.utils.output_formatter import formatter
    
    # Create output directory
    output_dir = os.path.join(scene_path, "color_adapted_materials", part_id)
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        "part_id": part_id,
        "target_rgb": list(target_rgb),
        "materials": []
    }
    
    for idx, material_id in enumerate(top_4_materials, 1):
        # Load albedo map
        albedo_path = load_material_albedo(material_id)
        
        if not albedo_path or not os.path.exists(albedo_path):
            formatter.print_warning(f"Albedo not found for {material_id}, skipping", indent=1)
            results["materials"].append({
                "material_id": material_id,
                "status": "failed",
                "error": "Albedo map not found"
            })
            continue
        
        # Load albedo image
        albedo_image = cv2.imread(albedo_path)
        if albedo_image is None:
            formatter.print_warning(f"Failed to load albedo for {material_id}", indent=1)
            results["materials"].append({
                "material_id": material_id,
                "status": "failed",
                "error": "Failed to load albedo image"
            })
            continue
        
        # Apply color adaptation
        try:
            adapted_image = apply_color_to_texture(albedo_image, target_rgb)
            
            # Resize to 400x400 and save as JPG (this is just cache for VLM selection)
            adapted_resized = cv2.resize(adapted_image, (400, 400), interpolation=cv2.INTER_AREA)
            output_path = os.path.join(output_dir, f"{material_id}_adapted.jpg")
            cv2.imwrite(output_path, adapted_resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            results["materials"].append({
                "material_id": material_id,
                "status": "success",
                "original_albedo": albedo_path,
                "adapted_albedo": output_path,
                "target_rgb": list(target_rgb)
            })

            # Removed verbose success logging - keeps terminal output clean
            
        except Exception as e:
            formatter.print_error(f"Error adapting color for {material_id}: {str(e)}", indent=1)
            results["materials"].append({
                "material_id": material_id,
                "status": "failed",
                "error": str(e)
            })
    
    # Save results
    result_file = os.path.join(output_dir, "color_adaptation_results.json")
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def color_adapt_image(scene_path, object_name, use_ot=False):
    """
    Main function to adapt colors of material textures.
    
    Args:
        scene_path: Path to the scene directory
        object_name: Name of the object to process
        use_ot: Whether to use Optimal Transport for color adaptation (default: False)
    """
    start_time = time.time()
    
    # ---------- LOAD COLOR DATA ----------
    # Try new format first (final_rgb_colors.json), then fallback to old format (rgb_only.json)
    # Note: object_name already includes "_gpt" suffix (e.g., "Chair_gpt")
    # scene_path is the object folder (e.g., output/mat_painting_stage/scene_name/Chair1)
    # So we should use object_name directly without appending _gpt again
    if object_name.endswith("_gpt"):
        object_gpt_folder = f"{scene_path}/{object_name}"
    else:
        object_gpt_folder = f"{scene_path}/{object_name}_gpt"
    final_rgb_file = f"{object_gpt_folder}/gpt4_query/final_rgb_colors.json"
    json_file = f"{object_gpt_folder}/gpt4_query/rgb_only.json"
    
    color_data = None
    
    # Try new format first
    if os.path.exists(final_rgb_file):
        try:
            with open(final_rgb_file) as f:
                final_colors = json.load(f)
            # Convert to old format: part_id -> "[R, G, B]" string
            color_data = {}
            for part_id, rgb_list in final_colors.items():
                if isinstance(rgb_list, list) and len(rgb_list) == 3:
                    color_data[part_id] = f"[{rgb_list[0]}, {rgb_list[1]}, {rgb_list[2]}]"
                else:
                    color_data[part_id] = str(rgb_list)
            formatter.print_success(f"Loaded color data from new format (final_rgb_colors.json) for {len(color_data)} parts")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            formatter.print_warning(f"Could not load new format, trying old format: {e}")
            color_data = None
    
    # Fallback to old format
    if color_data is None:
        try:
            with open(json_file) as f:
                color_data = json.load(f)
                formatter.print_success(f"Loaded color data from old format (rgb_only.json) for {len(color_data)} parts")
        except FileNotFoundError:
            formatter.print_error(f"Color data file not found at {json_file} or {final_rgb_file}")
            return
        except json.JSONDecodeError:
            formatter.print_error("Invalid JSON in color data file")
            return
    
    # ---------- PREPARE OUTPUT DIRECTORIES ----------
    # Source material folder - try visual retrieval first, then LLM fallback
    original_folder = f"{scene_path}/selected_material/{object_name}"
    fallback_folder = f"{scene_path}/Onboarded/decomposed/select_mat"
    
    if not os.path.exists(original_folder):
        # Visual retrieval failed, check if LLM fallback materials exist
        if os.path.exists(fallback_folder):
            formatter.print_warning(f"Visual retrieval materials not found at {original_folder}")
            formatter.print_info(f"Using LLM fallback materials from {fallback_folder}")
            # Use select_mat as source, but we'll copy to selected_material_OT_with_adaptation structure
            original_folder = fallback_folder
        else:
            formatter.print_error(f"No materials found (neither visual retrieval nor LLM fallback)")
            formatter.print_error(f"  Expected: {original_folder}")
            formatter.print_error(f"  Or: {fallback_folder}")
        return
        
    # Define target directory (skip OT_only - it's never used)
    with_adaptation_target = f"{scene_path}/selected_material_OT_with_adaptation/{object_name}"
    
    # Clear existing directory if it exists
    if os.path.exists(with_adaptation_target):
        shutil.rmtree(with_adaptation_target)
    
    # Copy original materials to target directory
    # If using fallback, we need to restructure: select_mat has parts directly, but we need object_name/part structure
    if original_folder == fallback_folder:
        # Create object_name subdirectory structure
        os.makedirs(with_adaptation_target, exist_ok=True)
        # Copy each part from select_mat to the new structure
        for part_name in os.listdir(fallback_folder):
            part_path = os.path.join(fallback_folder, part_name)
            if os.path.isdir(part_path):
                target_part_path = os.path.join(with_adaptation_target, part_name)
                shutil.copytree(part_path, target_part_path)
        formatter.print_success(f"LLM fallback materials copied to output directory")
    else:
        # Normal case: copy from selected_material
        shutil.copytree(original_folder, with_adaptation_target)
        formatter.print_success("Output directory prepared (skipped OT_only - unused)")
    
    # ---------- PROCESS EACH PART ----------
    total_parts = len(color_data)
    processed_parts = 0
    skipped_parts = 0
    
    # Determine if we're using fallback materials (select_mat) or normal materials (selected_material)
    using_fallback = (original_folder == fallback_folder)
    
    # Create refinement log
    refinement_log = {
        "method": "color_adaptation",
        "use_ot": use_ot,
        "parts": {}
    }
    
    formatter.print_info(f"Processing {total_parts} parts...")
    
    for part_idx, (part_name, color_str) in enumerate(color_data.items(), 1):
        part_start_time = time.time()
        
        # ---------- LOCATE REFERENCE IMAGES ----------
        reference_folder = f'{scene_path}/select_crop/{object_name}/{part_name}'
        
        # ---------- LOCATE TARGET TEXTURE ----------
        # Check both normal path and fallback path
        if using_fallback:
            # Fallback: materials are in select_mat/{part_name}/
            target_image_path_pattern = f'{scene_path}/Onboarded/decomposed/select_mat/{part_name}/basecolor.png'
        else:
            # Normal: materials are in selected_material/{object_name}/{part_name}/
            target_image_path_pattern = f'{scene_path}/selected_material/{object_name}/{part_name}/basecolor.png'
        
        target_image_files = glob(target_image_path_pattern)
        
        if not target_image_files:
            skipped_parts += 1
            elapsed = time.time() - part_start_time
            formatter.print_warning(f"{part_name} → No target texture found ({elapsed:.1f}s)", indent=1)
            continue
            
        target_image_path = target_image_files[0]
        
        # ---------- LOAD TARGET TEXTURE ----------
        target_image = cv2.imread(target_image_path)
        if target_image is None:
            skipped_parts += 1
            elapsed = time.time() - part_start_time
            formatter.print_error(f"{part_name} → Could not load texture ({elapsed:.1f}s)", indent=1)
            continue
        
        # ---------- APPLY OPTIMAL TRANSPORT ----------
        if use_ot and os.path.exists(reference_folder):
            # Get color distribution from reference images
            source_colors = average_color_distribution(reference_folder)
            # Apply optimal transport color transfer
            adapted_image = apply_color_transfer(source_colors, target_image)
        else:
            adapted_image = target_image.copy()
        
        # ---------- APPLY COLOR ADJUSTMENT ----------
        try:
            # Parse RGB color from string
            rgb_color = tuple(map(int, color_str.strip("[]").split(", ")))
            
            # Validate RGB values are in valid range
            if len(rgb_color) != 3 or any(not (0 <= val <= 255) for val in rgb_color):
                raise ValueError(f"Invalid RGB values: {rgb_color} - values must be between 0 and 255")
            
            # Optional: Sanity check for furniture colors (warn if suspicious)
            r, g, b = rgb_color
            # Check if color is suspiciously saturated/vibrant for furniture
            max_val = max(r, g, b)
            min_val = min(r, g, b)
            saturation = (max_val - min_val) / max_val if max_val > 0 else 0
            
            # If very saturated and high blue (purple), it's likely wrong
            if saturation > 0.6 and b > 150 and b > r and b > g:
                formatter.print_warning(f"Suspicious color detected for {part_name}: {rgb_color} - high saturation purple/violet. This may be incorrect.", indent=1)
            
            # Apply color adjustment
            final_image = apply_color_to_texture(target_image, rgb_color)
        except Exception as e:
            formatter.print_error(f"Error parsing RGB color '{color_str}' for {part_name}: {e}. Using original texture.", indent=1)
            final_image = target_image.copy()
        
        # ---------- SAVE RESULTS ----------
        # Save final results (OT + color adjustment) - skip OT_only since it's unused
        final_basecolor_path = f'{with_adaptation_target}/{part_name}/basecolor.png'
        final_diffuse_path = f'{with_adaptation_target}/{part_name}/diffuse.png'
        cv2.imwrite(final_basecolor_path, final_image)
        cv2.imwrite(final_diffuse_path, final_image)
        
        # Log before/after information
        before_basecolor = target_image_path
        after_basecolor = final_basecolor_path
        
        refinement_log["parts"][part_name] = {
            "target_rgb": color_str,
            "target_rgb_parsed": list(rgb_color) if 'rgb_color' in locals() else None,
            "before_material": {
                "basecolor": before_basecolor
            },
            "after_material": {
                "basecolor": after_basecolor
            },
            "processing_time_seconds": time.time() - part_start_time
        }
        
        processed_parts += 1
        part_time = time.time() - part_start_time
        formatter.print_success(f"{part_name} → Color adapted (RGB: {rgb_color if 'rgb_color' in locals() else 'N/A'}) ({part_time:.1f}s)", indent=1)
    
    # ---------- SUMMARY ----------
    total_time = time.time() - start_time
    
    # Save refinement log
    refinement_log["total_processing_time_seconds"] = total_time
    refinement_log["total_parts"] = total_parts
    refinement_log["processed_parts"] = processed_parts
    refinement_log["skipped_parts"] = skipped_parts
    
    log_file = f"{scene_path}/material_refinement_log.json"
    with open(log_file, 'w') as f:
        json.dump(refinement_log, f, indent=2)
    
    formatter.print_summary(
        f"Material refinement complete for: {object_name}",
        {
            "Total parts": total_parts,
            "Successfully processed": processed_parts,
            "Skipped": skipped_parts
        },
        total_time
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply color adaptation to material textures.")
    parser.add_argument("--scene", type=str, required=True,
                        help="Path to the folder containing scene data.")
    parser.add_argument("--object", type=str, help="Object name (optional, will be inferred from scene path if not provided)")
    parser.add_argument("--use_ot", action='store_true',
                        help="Whether to use Optimal Transport for color adaptation (default: False)")
    args = parser.parse_args()
    
    scene_path = args.scene
    
    if args.object:
        object_name = args.object
    else:
        # Extract object name and clean it
        obj_name = scene_path.split("/")[-1]
        semantic_name = ''.join(filter(lambda x: not x.isdigit(), obj_name))
        
        # Apply same naming logic as apply_mat_rotate.py for windows/doors
        if "Window" not in semantic_name and "Door" not in semantic_name:
            semantic_name = semantic_name.replace("_", "").replace("Wall", "")
            object_name = semantic_name + "_gpt"
        else:
            # Windows and Doors use special naming: "Wall_" + semantic_name + "__gpt"
            semantic_name = semantic_name.replace("_", "")
            semantic_name = semantic_name.replace("Wall", "")
            if "Window" in semantic_name:
                object_name = "Wall_" + semantic_name + "__gpt"
            elif "Door" in semantic_name:
                object_name = "Wall_" + semantic_name + "__gpt"
    
    print(f"🔍 Processing scene: {scene_path}")
    print(f"🔍 Object name: {object_name}")
    print(f"🔍 Using Optimal Transport: {'Yes' if args.use_ot else 'No'}")
    
    color_adapt_image(scene_path, object_name, args.use_ot)
