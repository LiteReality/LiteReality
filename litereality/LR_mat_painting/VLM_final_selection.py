"""
VLM Final Material Selection

Phase 4: Selects the best material from 4 color-adapted candidates based on
complete visual analysis.

This module uses VLM (Vision-Language Models) to make the final material selection
by comparing color-adapted material textures against the reference images.
"""

import os
import json
import cv2
import numpy as np
import time
import shutil
from litereality.LR_mat_painting.utils.qwen_query_materials import process_image_qwen
from litereality.LR_mat_painting.utils.vlm_logger import log_vlm_conversation, prepare_image_info
from litereality.LR_mat_painting.utils.output_formatter import formatter
from litereality.LR_mat_painting.Material_refinements import load_material_albedo
from litereality.LR_mat_painting.prompts import get_final_selection_prompt


def create_final_comparison_layout(reference_image_path, part_rendering_path, color_adapted_paths, output_path=None):
    """
    Create comparison layout for final selection.
    
    Layout:
    [Reference Image] [3D Rendering]
    [Material 1 Adapted] [Material 2 Adapted] [Material 3 Adapted] [Material 4 Adapted]
    
    Args:
        reference_image_path: Path to reference image
        part_rendering_path: Path to 3D part rendering
        color_adapted_paths: List of 4 paths to color-adapted materials
        output_path: Optional path to save the layout image
    
    Returns:
        str: Path to saved comparison image
    """
    # Load images
    ref_img = cv2.imread(reference_image_path) if reference_image_path and os.path.exists(reference_image_path) else None
    render_img = cv2.imread(part_rendering_path) if os.path.exists(part_rendering_path) else None
    
    # Resize to consistent size
    material_size = (200, 200)
    top_row_height = 300
    
    if ref_img is not None:
        ref_img = cv2.resize(ref_img, (300, top_row_height))
    else:
        ref_img = np.ones((top_row_height, 300, 3), dtype=np.uint8) * 255
    
    if render_img is not None:
        render_img = cv2.resize(render_img, (300, top_row_height))
    else:
        render_img = np.ones((top_row_height, 300, 3), dtype=np.uint8) * 255
    
    # Load and resize color-adapted materials
    material_images = []
    for adapted_path in color_adapted_paths[:4]:  # Limit to 4
        if adapted_path and os.path.exists(adapted_path):
            mat_img = cv2.imread(adapted_path)
            if mat_img is not None:
                resized = cv2.resize(mat_img, material_size)
                material_images.append(resized)
            else:
                material_images.append(np.ones(material_size + (3,), dtype=np.uint8) * 128)
        else:
            material_images.append(np.ones(material_size + (3,), dtype=np.uint8) * 128)
    
    # Pad to 4 if needed
    while len(material_images) < 4:
        material_images.append(np.ones(material_size + (3,), dtype=np.uint8) * 255)
    
    # Create top row
    top_row = np.hstack([ref_img, render_img])
    
    # Create bottom row with 4 materials
    bottom_row = np.hstack(material_images)
    
    # Combine
    comparison_width = max(top_row.shape[1], bottom_row.shape[1])
    
    # Pad if needed
    if top_row.shape[1] < comparison_width:
        pad_width = comparison_width - top_row.shape[1]
        top_row = np.hstack([top_row, np.ones((top_row.shape[0], pad_width, 3), dtype=np.uint8) * 255])
    
    comparison_image = np.vstack([top_row, bottom_row])
    
    # Save if output path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, comparison_image)
    
    return comparison_image


def parse_final_selection_response(response_content, material_ids):
    """
    Parse VLM response to extract selected material.
    
    Args:
        response_content: Raw response content from VLM
        material_ids: List of material IDs in order
    
    Returns:
        str: Selected material ID, or None if parsing failed
    """
    import re
    
    # Try to extract JSON
    json_match = re.search(r'\{[\s\S]*\}', response_content)
    if json_match:
        try:
            json_str = json_match.group(0)
            parsed = json.loads(json_str)
            
            if "selected_material" in parsed:
                return parsed["selected_material"]
        except json.JSONDecodeError:
            pass
    
    # Try to find material ID in text
    for mat_id in material_ids:
        if mat_id.lower() in response_content.lower():
            return mat_id
    
    # Try to extract by index (e.g., "material 1", "first material")
    index_patterns = [
        r'material\s*(\d+)',
        r'(\d+)(?:st|nd|rd|th)\s*material',
        r'option\s*(\d+)',
    ]
    
    for pattern in index_patterns:
        match = re.search(pattern, response_content.lower())
        if match:
            idx = int(match.group(1)) - 1
            if 0 <= idx < len(material_ids):
                return material_ids[idx]
    
    return None


def select_final_material(scene_path, part_id, color_adapted_materials, object_type="object"):
    """
    Phase 4: Select final material from 4 color-adapted candidates.
    
    Args:
        scene_path: Path to scene folder
        part_id: Part identifier (e.g., "solid_001")
        color_adapted_materials: List of dicts with material info from Phase 3
        object_type: Object type (e.g., "Chair")
    
    Returns:
        dict: Final selection result
    """
    # Get paths
    object_name_clean = ''.join(filter(lambda x: not x.isdigit(), scene_path.split("/")[-1]))
    folder_path = f"{scene_path}/{object_name_clean}_gpt"
    
    reference_image_path = os.path.join(scene_path, "captured_images", "stitched_image.jpg")
    part_rendering_path = os.path.join(scene_path, "Onboarded", "top_4_combine", f"{part_id}.png")
    
    # Extract material IDs and adapted paths
    material_ids = []
    adapted_paths = []
    
    for mat_info in color_adapted_materials:
        if mat_info.get("status") == "success":
            material_ids.append(mat_info["material_id"])
            adapted_paths.append(mat_info["adapted_albedo"])
    
    if len(material_ids) < 4:
        formatter.print_warning(f"Only {len(material_ids)} successful color adaptations for {part_id}", indent=1)
        if len(material_ids) == 0:
            return {
                "selected_material": None,
                "status": "failed",
                "error": "No successful color adaptations"
            }
    
    # Create comparison layout
    comparison_output_path = os.path.join(folder_path, "vlm_conversations", f"final_selection_{part_id}.jpg")
    comparison_image = create_final_comparison_layout(
        reference_image_path, part_rendering_path, adapted_paths, comparison_output_path
    )
    
    # Create prompt for final selection using centralized prompt generator
    final_prompt = get_final_selection_prompt(material_ids, object_type)
    
    # Prepare input images for logging
    input_images = [
        prepare_image_info(comparison_output_path, "Final selection comparison with reference, 3D rendering, and 4 color-adapted materials", "final_comparison")
    ]
    if reference_image_path and os.path.exists(reference_image_path):
        input_images.append(
            prepare_image_info(reference_image_path, f"Reference image showing {object_type}", "reference_image")
        )
    
    # Query VLM
    count = 0
    success = False
    selected_material = None
    parsing_errors = []
    raw_response = None
    confidence = None
    reasoning = ""
    
    while count < 5:
        count += 1
        try:
            start_time = time.time()
            response = process_image_qwen(comparison_output_path, final_prompt, max_new_tokens=500, max_image_size=1200)
            processing_time = time.time() - start_time
            
            raw_response = response
            content = response['choices'][0]['message']['content']
            
            # Parse response
            selected_material = parse_final_selection_response(content, material_ids)
            
            if selected_material and selected_material in material_ids:
                success = True
                # Try to extract confidence and reasoning
                import re
                json_match = re.search(r'\{[\s\S]*\}', content)
                if json_match:
                    try:
                        parsed = json.loads(json_match.group(0))
                        confidence = parsed.get("confidence")
                        reasoning = parsed.get("reasoning", "")
                    except:
                        pass
                break
            else:
                parsing_errors.append(f"Invalid material selection: {selected_material}")
                
        except Exception as e:
            parsing_errors.append(f"Error on attempt {count}: {str(e)}")
            time.sleep(2)
            continue
    
    # Fallback if parsing failed
    if not success or not selected_material:
        formatter.print_warning(f"Final selection parsing failed for {part_id}, using first material", indent=1)
        selected_material = material_ids[0] if material_ids else None
        reasoning = "Fallback: first material (parsing failed)"
    
    # Log conversation
    log_vlm_conversation(
        scene_path=scene_path,
        phase="phase_4_final_selection",
        part_id=part_id,
        object_name=object_name_clean,
        prompt=final_prompt,
        input_images=input_images,
        raw_response=raw_response or {},
        parsed_result={
            "selected_material": selected_material,
            "confidence": confidence,
            "reasoning": reasoning
        },
        metadata={
            "num_candidates": len(material_ids),
            "comparison_image_path": comparison_output_path
        },
        prompt_template="final_selection",
        parsing_errors=parsing_errors,
        success=success,
        error_message=None if success else "Failed to parse final selection"
    )
    
    # Copy selected material to output directory
    selected_material_path = None
    if selected_material:
        # Find the material info
        for mat_info in color_adapted_materials:
            if mat_info.get("material_id") == selected_material:
                # Copy to selected_material folder
                object_name = object_name_clean + "_gpt"
                output_folder = os.path.join(scene_path, "selected_material", object_name, part_id)
                os.makedirs(output_folder, exist_ok=True)
                
                # Copy all PBR maps from original material
                original_albedo = load_material_albedo(selected_material)
                if original_albedo:
                    material_dir = os.path.dirname(original_albedo)
                    
                    # Copy all texture maps
                    texture_maps = ["basecolor.png", "diffuse.png", "normal.png", "roughness.png", 
                                   "metallic.png", "height.png", "displacement.png", "specular.png"]
                    
                    for tex_map in texture_maps:
                        src_path = os.path.join(material_dir, tex_map)
                        if os.path.exists(src_path):
                            dst_path = os.path.join(output_folder, tex_map)
                            shutil.copy2(src_path, dst_path)
                    
                    # Overwrite basecolor/diffuse with color-adapted version
                    adapted_path = mat_info.get("adapted_albedo")
                    if adapted_path and os.path.exists(adapted_path):
                        cv2.imwrite(os.path.join(output_folder, "basecolor.png"), cv2.imread(adapted_path))
                        cv2.imwrite(os.path.join(output_folder, "diffuse.png"), cv2.imread(adapted_path))
                    
                    selected_material_path = output_folder
    
    # Save result
    result = {
        "part_id": part_id,
        "selected_material": selected_material,
        "confidence": confidence,
        "reasoning": reasoning,
        "material_path": selected_material_path,
        "status": "success" if success else "failed",
        "prompt": final_prompt  # Include the prompt used for transparency
    }
    
    result_file = os.path.join(folder_path, "gpt4_query", "final_material_selection.json")
    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    
    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            results = json.load(f)
    else:
        results = {}
    
    results[part_id] = result
    
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    formatter.print_success(f"{part_id} → Final material selected: {selected_material}", indent=1)
    
    return result
