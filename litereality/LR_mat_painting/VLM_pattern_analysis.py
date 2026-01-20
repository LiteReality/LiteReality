"""
VLM Pattern Analysis for Material Selection

Phase 2: Analyzes albedo patterns of top 10 materials to narrow down to top 4
best pattern matches.

This module uses VLM (Vision-Language Models) to compare texture patterns between
material candidates and reference images, selecting the materials with the most
similar visual patterns.
"""

import os
import json
import cv2
import numpy as np
import time
from pathlib import Path
from litereality.LR_mat_painting.utils.qwen_query_materials import process_image_qwen
from litereality.LR_mat_painting.utils.vlm_logger import log_vlm_conversation, prepare_image_info
from litereality.LR_mat_painting.utils.output_formatter import formatter
from litereality.LR_mat_painting.prompts import get_pattern_analysis_prompt


def load_material_albedo(material_id):
    """
    Load albedo/basecolor map for a material.

    Args:
        material_id: Material ID (e.g., "Plastic_Plastic015A")

    Returns:
        str: Path to albedo map, or None if not found
    """
    # Use the same approach as the old link_pbr_materials function
    from litereality.LR_mat_painting.utils.gpt4_query_materials import format_string

    # Parse material ID similar to old approach
    parts = material_id.split("_")
    if len(parts) < 2:
        return None

    category = parts[0]  # "Plastic"
    full_name = "_".join(parts[1:])  # Join all parts after category (e.g., "Plastic015A" or "acg_chip_001")

    base_path = "litereality_database/PBR_materials/material_lib/pbr_maps/train"

    # Try multiple path patterns (similar to Material_refinements.py)
    # 1. Direct name (for ACG-style names like "acg_chip_001")
    # 2. Formatted name (for old-style names like "Plastic015A" -> "acg_plastic_015_a")
    possible_dir_names = [full_name]

    # Only add formatted version if name doesn't already start with "acg_"
    if not full_name.startswith("acg_"):
        possible_dir_names.append(format_string(full_name))

    for dir_name in possible_dir_names:
        basecolor_path = f"{base_path}/{category}/{dir_name}/basecolor.png"
        if os.path.exists(basecolor_path):
            return basecolor_path

        diffuse_path = f"{base_path}/{category}/{dir_name}/diffuse.png"
        if os.path.exists(diffuse_path):
            return diffuse_path

    print(f"Warning: Material {material_id} not found at {base_path}/{category}/{full_name}/basecolor.png or diffuse.png")
    return None


def create_pattern_comparison_layout(reference_image_path, part_rendering_path, albedo_paths, output_path=None):
    """
    Create comparison layout image for VLM pattern analysis.
    
    Layout:
    [Reference Image] [3D Rendering]
    [Albedo 1] [Albedo 2] [Albedo 3] ... [Albedo 20]
    
    Args:
        reference_image_path: Path to reference image
        part_rendering_path: Path to 3D part rendering
        albedo_paths: List of paths to albedo maps (up to 20)
        output_path: Optional path to save the layout image
    
    Returns:
        str: Path to saved comparison image
    """
    # Load images
    ref_img = cv2.imread(reference_image_path) if reference_image_path and os.path.exists(reference_image_path) else None
    render_img = cv2.imread(part_rendering_path) if os.path.exists(part_rendering_path) else None
    
    # Resize to consistent size
    thumb_size = (150, 150)
    top_row_height = 300
    
    if ref_img is not None:
        ref_img = cv2.resize(ref_img, (300, top_row_height))
    else:
        ref_img = np.ones((top_row_height, 300, 3), dtype=np.uint8) * 255
    
    if render_img is not None:
        render_img = cv2.resize(render_img, (300, top_row_height))
    else:
        render_img = np.ones((top_row_height, 300, 3), dtype=np.uint8) * 255
    
    # Load and resize albedo maps
    albedo_thumbnails = []
    for albedo_path in albedo_paths[:20]:  # Limit to 20
        if albedo_path and os.path.exists(albedo_path):
            albedo = cv2.imread(albedo_path)
            if albedo is not None:
                thumb = cv2.resize(albedo, thumb_size)
                albedo_thumbnails.append(thumb)
            else:
                # Placeholder if image failed to load
                albedo_thumbnails.append(np.ones(thumb_size + (3,), dtype=np.uint8) * 128)
        else:
            # Placeholder if path doesn't exist
            albedo_thumbnails.append(np.ones(thumb_size + (3,), dtype=np.uint8) * 128)
    
    # Create top row
    top_row = np.hstack([ref_img, render_img])
    
    # Create bottom row(s) - arrange in grid
    cols_per_row = 5
    rows_needed = (len(albedo_thumbnails) + cols_per_row - 1) // cols_per_row
    
    bottom_rows = []
    for row_idx in range(rows_needed):
        start_idx = row_idx * cols_per_row
        end_idx = min(start_idx + cols_per_row, len(albedo_thumbnails))
        row_thumbnails = albedo_thumbnails[start_idx:end_idx]
        
        # Pad row if needed
        while len(row_thumbnails) < cols_per_row:
            row_thumbnails.append(np.ones(thumb_size + (3,), dtype=np.uint8) * 255)
        
        row = np.hstack(row_thumbnails)
        bottom_rows.append(row)
    
    bottom_section = np.vstack(bottom_rows) if bottom_rows else np.ones((thumb_size[1], thumb_size[0] * cols_per_row, 3), dtype=np.uint8) * 255
    
    # Combine top and bottom
    comparison_width = max(top_row.shape[1], bottom_section.shape[1])
    
    # Pad top row if needed
    if top_row.shape[1] < comparison_width:
        pad_width = comparison_width - top_row.shape[1]
        top_row = np.hstack([top_row, np.ones((top_row.shape[0], pad_width, 3), dtype=np.uint8) * 255])
    
    comparison_image = np.vstack([top_row, bottom_section])
    
    # Save if output path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, comparison_image)
    
    return comparison_image


def parse_pattern_analysis_response(response_content):
    """
    Parse VLM response to extract top 4 materials.
    
    Args:
        response_content: Raw response content from VLM
    
    Returns:
        dict: Parsed result with top_4 materials and reasoning
    """
    import re
    
    # Try to extract JSON
    json_match = re.search(r'\{[\s\S]*\}', response_content)
    if json_match:
        try:
            json_str = json_match.group(0)
            parsed = json.loads(json_str)
            
            if "top_4" in parsed:
                return {
                    "top_4": parsed["top_4"],
                    "scores": parsed.get("scores", {}),
                    "reasoning": parsed.get("reasoning", "")
                }
        except json.JSONDecodeError:
            pass
    
    # Fallback: try to extract material IDs from text
    material_ids = re.findall(r'([A-Za-z_]+[0-9]+[A-Za-z]*)', response_content)
    if len(material_ids) >= 4:
        return {
            "top_4": material_ids[:4],
            "scores": {mat_id: 0.5 for mat_id in material_ids[:4]},
            "reasoning": "Extracted from text response"
        }
    
    return None


def analyze_albedo_patterns(scene_path, part_id, top_10_materials, object_type="object"):
    """
    Analyze albedo patterns to select top 4 materials.
    
    Args:
        scene_path: Path to scene folder
        part_id: Part identifier (e.g., "solid_001")
        top_20_materials: List of top 20 material IDs from Phase 1
        object_type: Object type (e.g., "Chair")
    
    Returns:
        dict: Top 4 materials with pattern analysis results
    """
    # Get paths
    object_name_clean = ''.join(filter(lambda x: not x.isdigit(), scene_path.split("/")[-1]))
    folder_path = f"{scene_path}/{object_name_clean}_gpt"
    
    reference_image_path = os.path.join(scene_path, "captured_images", "stitched_image.jpg")
    part_rendering_path = os.path.join(scene_path, "Onboarded", "top_4_combine", f"{part_id}.png")
    
    # Load albedo maps for top 20 materials
    albedo_paths = []
    valid_materials = []
    for mat_id in top_10_materials:
        albedo_path = load_material_albedo(mat_id)
        if albedo_path:
            albedo_paths.append(albedo_path)
            valid_materials.append(mat_id)
        else:
            formatter.print_warning(f"Albedo not found for {mat_id}, skipping", indent=1)
    
    if len(valid_materials) < 4:
        formatter.print_error(f"Not enough valid materials ({len(valid_materials)}) for pattern analysis")
        return {
            "top_4": valid_materials[:4] if len(valid_materials) >= 4 else valid_materials,
            "scores": {},
            "reasoning": "Insufficient materials for analysis"
        }
    
    # Create comparison layout
    comparison_output_path = os.path.join(folder_path, "vlm_conversations", f"pattern_comparison_{part_id}.jpg")
    comparison_image = create_pattern_comparison_layout(
        reference_image_path, part_rendering_path, albedo_paths, comparison_output_path
    )
    
    # Create prompt for pattern analysis using centralized prompt generator
    pattern_prompt = get_pattern_analysis_prompt(valid_materials, object_type)
    
    # Prepare input images for logging
    input_images = [
        prepare_image_info(comparison_output_path, "Pattern comparison layout with reference, 3D rendering, and 20 albedo maps", "pattern_comparison")
    ]
    if reference_image_path and os.path.exists(reference_image_path):
        input_images.append(
            prepare_image_info(reference_image_path, f"Reference image showing {object_type}", "reference_image")
        )
    
    # Query VLM
    count = 0
    success = False
    parsed_result = None
    parsing_errors = []
    raw_response = None
    
    while count < 5:
        count += 1
        try:
            start_time = time.time()
            response = process_image_qwen(comparison_output_path, pattern_prompt, max_new_tokens=500, max_image_size=1200)
            processing_time = time.time() - start_time
            
            raw_response = response
            content = response['choices'][0]['message']['content']
            
            # Parse response
            parsed_result = parse_pattern_analysis_response(content)
            
            if parsed_result and "top_4" in parsed_result and len(parsed_result["top_4"]) == 4:
                success = True
                break
            else:
                parsing_errors.append(f"Invalid parsed result: {parsed_result}")
                
        except Exception as e:
            parsing_errors.append(f"Error on attempt {count}: {str(e)}")
            time.sleep(2)
            continue
    
    # Log conversation
    log_vlm_conversation(
        scene_path=scene_path,
        phase="phase_2_pattern_analysis",
        part_id=part_id,
        object_name=object_name_clean,
        prompt=pattern_prompt,
        input_images=input_images,
        raw_response=raw_response or {},
        parsed_result=parsed_result,
        metadata={
            "num_materials_analyzed": len(valid_materials),
            "comparison_image_path": comparison_output_path
        },
        prompt_template="pattern_analysis",
        parsing_errors=parsing_errors,
        success=success,
        error_message=None if success else "Failed to parse top 4 pattern matches"
    )
    
    # Fallback if parsing failed
    if not success or not parsed_result:
        formatter.print_warning(f"Pattern analysis parsing failed for {part_id}, using first 4 materials", indent=1)
        parsed_result = {
            "top_4": valid_materials[:4],
            "scores": {mat_id: 0.5 for mat_id in valid_materials[:4]},
            "reasoning": "Fallback: first 4 materials (parsing failed)"
        }
    
    # Save result
    result_file = os.path.join(folder_path, "gpt4_query", "top_4_pattern_matched.json")
    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    
    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            results = json.load(f)
    else:
        results = {}
    
    results[part_id] = parsed_result
    
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    return {
        **parsed_result,
        "prompt": pattern_prompt  # Include the prompt used for transparency
    }
