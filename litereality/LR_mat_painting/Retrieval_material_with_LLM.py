"""
Material Retrieval System using LLM (Qwen3-VL)

This module orchestrates the complete material selection pipeline using VLM
(Vision-Language Models). It processes scene objects through multiple phases:

1. Part Name Identification - Identify and describe each segmented part
2. Holistic Color Analysis - Analyze overall object colors from multiple views
3. Enhanced Part Identification - Detailed part analysis with material hints
4. Color Voting - Extract RGB colors through multi-image voting
5. Color Validation - Validate and refine colors against holistic analysis
6. Main Material Type - Determine primary material categories
7. Secondary Categories - Refine material subcategories
8. Top 10 Selection - Select best material candidates from database
9. VLM Material Pipeline - Final selection through pattern analysis and color adaptation

Standard procedural version with hard-stop validation.
"""

import os
import json
import argparse
import time
import cv2
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

from litereality.LR_mat_painting.utils.gpt4_query_materials import (
    query_part_color, query_second_categories, query_part_merged,
    query_main_type, query_part_name,
    reduce_materail_size, query_holistic_colors, query_part_identification_enhanced,
    query_part_color_with_voting, validate_and_refine_colors, query_top_10_materials,
    create_rgb_only_compatibility
)
from litereality.LR_mat_painting.VLM_pattern_analysis import analyze_albedo_patterns, load_material_albedo
from litereality.LR_mat_painting.Material_refinements import color_adapt_multiple_materials
from litereality.LR_mat_painting.VLM_final_selection import select_final_material
from litereality.LR_mat_painting.utils.qwen_model_manager import ensure_model_loaded, unload_model_if_loaded

load_dotenv()

# ANSI color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def save_albedo_image(material_id, source_path, cache_dir, phase_name, part_id, size=(400, 400)):
    """
    Save a resized albedo image to the cache folder.
    
    Args:
        material_id: Material ID (e.g., "Plastic_Plastic015A")
        source_path: Path to source albedo image
        cache_dir: Cache directory path
        phase_name: Phase name (e.g., "phase_1_top_10")
        part_id: Part identifier
        size: Target size (width, height)
    
    Returns:
        str: Relative path to saved image, or None if failed
    """
    if not source_path or not os.path.exists(source_path):
        return None
    
    try:
        # Create phase directory
        phase_dir = os.path.join(cache_dir, "images", phase_name, part_id)
        os.makedirs(phase_dir, exist_ok=True)
        
        # Load and resize image
        img = cv2.imread(source_path)
        if img is None:
            return None
        
        # Resize maintaining aspect ratio, then crop/pad to exact size
        h, w = img.shape[:2]
        target_w, target_h = size
        
        # Calculate scaling to fit
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create canvas and center the image
        canvas = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        # Sanitize material_id for filename
        safe_material_id = material_id.replace("/", "_").replace("\\", "_")
        output_path = os.path.join(phase_dir, f"{safe_material_id}.jpg")
        
        # Save as JPEG with good quality
        cv2.imwrite(output_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
        
        # Return relative path from cache_dir
        rel_path = os.path.relpath(output_path, cache_dir)
        return rel_path.replace("\\", "/")  # Normalize path separators
        
    except Exception as e:
        print(f"⚠️ Warning: Failed to save albedo image for {material_id}: {e}")
        return None


def save_material_selection_cache(scene_path, object_name, material_selections):
    """
    Save material selection cache with all intermediate results for visualization.

    Args:
        scene_path: Path to scene folder
        object_name: Object name (e.g., "Chair_gpt")
        material_selections: Dict containing all selection phases data
    """
    # Create cache directory (already exists, but ensure it)
    cache_dir = material_selections.get("cache_directory", f"{scene_path}/material_selection_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Add summary information
    summary = {
        "total_parts": len(material_selections.get("parts", {})),
        "parts_summary": {}
    }
    
    for part_id, part_data in material_selections.get("parts", {}).items():
        summary["parts_summary"][part_id] = {
            "final_material": part_data.get("final_selected_material"),
            "target_color": part_data.get("target_color"),
            "num_top_10": len(part_data.get("top_10_materials", [])),
            "num_top_4": len(part_data.get("top_4_materials", [])),
            "num_adapted": len(part_data.get("adapted_materials", []))
        }
    
    material_selections["summary"] = summary

    # Save the complete selection data
    cache_file = f"{cache_dir}/selections.json"
    with open(cache_file, 'w') as f:
        json.dump(material_selections, f, indent=2)

    print(f"\n💾 Material selection cache saved:")
    print(f"   📄 JSON: {cache_file}")
    print(f"   🖼️  Images: {cache_dir}/images/")
    print(f"   📊 Summary: {summary['total_parts']} parts processed")


def create_material_selection_summary(part_id, phase_data):
    """
    Create a structured summary for a part's material selection process.

    Args:
        part_id: Part identifier (e.g., "solid_001")
        phase_data: Dict containing all phases data for this part

    Returns:
        dict: Structured summary
    """
    return {
        "part_id": part_id,
        "final_selected_material": phase_data.get("final_selected_material"),
        "selection_summary": {
            "top_10_materials": phase_data.get("top_10_materials", []),
            "top_4_materials": phase_data.get("top_4_materials", []),
            "adapted_materials": phase_data.get("adapted_materials", []),
            "final_material": phase_data.get("final_selected_material")
        },
        "prompts": {
            "phase_1_top_10": phase_data.get("prompts", {}).get("phase_1_top_10", ""),
            "phase_2_pattern_analysis": phase_data.get("prompts", {}).get("phase_2_pattern_analysis", ""),
            "phase_3_color_adaptation": phase_data.get("prompts", {}).get("phase_3_color_adaptation", ""),
            "phase_4_final_selection": phase_data.get("prompts", {}).get("phase_4_final_selection", "")
        },
        "metadata": {
            "target_color": phase_data.get("target_color"),
            "object_type": phase_data.get("object_type"),
            "scene_path": phase_data.get("scene_path")
        }
    }


def select_material_required_queries(scene_path):
    # --- INITIALIZATION ---
    # Reset model state to ensure clean loading
    from litereality.LR_mat_painting.utils.qwen_model_manager import reset_model_state
    reset_model_state()

    init_start = time.time()
    print(f"  {Colors.BLUE}🤖{Colors.RESET} Loading Qwen3-VL model for material selection...")
    try:
        ensure_model_loaded()
        init_elapsed = time.time() - init_start
        print(f"  {Colors.GREEN}✓{Colors.RESET} Model loaded successfully ({Colors.CYAN}{init_elapsed:.2f}s{Colors.RESET})")
    except RuntimeError as e:
        print(f"  {Colors.RED}❌ CRITICAL:{Colors.RESET} Model loading failed: {e}")
        print(f"  This is likely due to PyTorch version incompatibility.")
        print(f"  The pipeline cannot continue without the Qwen model.")
        raise

    api_key = None

    # --- PATH SETUP ---
    object_name_raw = scene_path.split("/")[-1]
    object_name = ''.join(filter(lambda x: not x.isdigit(), object_name_raw))
    type_str = object_name
    
    folder_path = f"{scene_path}/{object_name}_gpt"
    reference_image_path = f"{scene_path}/captured_images"
    reference_image = os.path.join(reference_image_path, "stitched_image.jpg")
    top_9_folder = f"{scene_path}/Onboarded/top_9_labeled_images"
    part_folder = f"{scene_path}/Onboarded/top_4_combine"
    material_output_path = f"{scene_path}/Onboarded/decomposed/select_mat"

    # --- STEP-BY-STEP EXECUTION ---
    step_start = time.time()
    print(f"\n{Colors.BOLD}{Colors.BLUE}[Step 1/9]{Colors.RESET} Identifying Part Names...")
    query_part_name(folder_path, part_folder, api_key, type_str=type_str)
    print(f"  {Colors.GREEN}✓{Colors.RESET} Part names identified ({Colors.CYAN}{time.time()-step_start:.2f}s{Colors.RESET})")

    step_start = time.time()
    print(f"\n{Colors.BOLD}{Colors.BLUE}[Step 2/9]{Colors.RESET} Performing Holistic Color Analysis...")
    holistic_colors = query_holistic_colors(folder_path, reference_image_path, api_key, type_str=type_str)
    print(f"  {Colors.GREEN}✓{Colors.RESET} Holistic color analysis complete ({Colors.CYAN}{time.time()-step_start:.2f}s{Colors.RESET})")

    step_start = time.time()
    print(f"\n{Colors.BOLD}{Colors.BLUE}[Step 3/9]{Colors.RESET} Running Enhanced Part Identification...")
    part_id_info = query_part_identification_enhanced(folder_path, part_folder, api_key, type_str=type_str)
    print(f"  {Colors.GREEN}✓{Colors.RESET} Part identification complete ({Colors.CYAN}{time.time()-step_start:.2f}s{Colors.RESET})")

    step_start = time.time()
    print(f"\n{Colors.BOLD}{Colors.BLUE}[Step 4/9]{Colors.RESET} Analyzing Colors with Voting...")
    query_part_color_with_voting(folder_path, reference_image, top_9_folder, part_id_info, holistic_colors, api_key, type_str=type_str)
    print(f"  {Colors.GREEN}✓{Colors.RESET} Color voting complete ({Colors.CYAN}{time.time()-step_start:.2f}s{Colors.RESET})")

    step_start = time.time()
    print(f"\n{Colors.BOLD}{Colors.BLUE}[Step 5/9]{Colors.RESET} Validating and Refining Colors...")
    validate_and_refine_colors(folder_path, f"{folder_path}/gpt4_query/holistic_colors.json", 
                               f"{folder_path}/gpt4_query/rgb_voting_results.json", reference_image, 
                               top_9_folder, part_id_info, api_key, type_str=type_str)
    create_rgb_only_compatibility(folder_path)
    print(f"  {Colors.GREEN}✓{Colors.RESET} Color validation complete ({Colors.CYAN}{time.time()-step_start:.2f}s{Colors.RESET})")

    step_start = time.time()
    print(f"\n{Colors.BOLD}{Colors.BLUE}[Step 6/9]{Colors.RESET} Determining Main Material Types...")
    query_main_type(folder_path, api_key, part_folder, type_str)
    print(f"  {Colors.GREEN}✓{Colors.RESET} Main material types determined ({Colors.CYAN}{time.time()-step_start:.2f}s{Colors.RESET})")

    step_start = time.time()
    print(f"\n{Colors.BOLD}{Colors.BLUE}[Step 7/9]{Colors.RESET} Refining Secondary Categories...")
    query_second_categories(folder_path, api_key, part_folder, type_str)
    print(f"  {Colors.GREEN}✓{Colors.RESET} Secondary categories refined ({Colors.CYAN}{time.time()-step_start:.2f}s{Colors.RESET})")

    step_start = time.time()
    print(f"\n{Colors.BOLD}{Colors.BLUE}[Step 8/9]{Colors.RESET} Selecting Top 10 Materials...")
    top_10_results = query_top_10_materials(folder_path, api_key, part_folder, type_str, scene_path)
    print(f"  {Colors.GREEN}✓{Colors.RESET} Top 10 materials selected ({Colors.CYAN}{time.time()-step_start:.2f}s{Colors.RESET})")

    # HARD STOP VALIDATION
    if not top_10_results:
        print(f"  {Colors.RED}❌ CRITICAL ERROR:{Colors.RESET} Top 10 material selection returned nothing. Pipeline stopped.")
        raise RuntimeError("CRITICAL ERROR: Top 10 material selection returned nothing. Pipeline stopped.")

    # unload_model_if_loaded()
    
    return {
        "folder_path": folder_path,
        "object_name": object_name,
        "stitched_image_path": part_folder,
        "type_str": type_str,
        "material_output_path": material_output_path,
        "api_key": api_key,
        "top_10_results": top_10_results,
        "scene_path": scene_path
    }

def run_vlm_material_pipeline(context):
    if not context: return

    scene_path = context.get("scene_path")
    folder_path = context.get("folder_path")
    type_str = context.get("type_str")
    top_10_results = context.get("top_10_results")

    if not all([scene_path, folder_path, type_str, top_10_results]):
        raise RuntimeError("CRITICAL: Missing required context information for VLM pipeline.")

    # Initialize material selections cache
    cache_dir = f"{scene_path}/material_selection_cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    material_selections = {
        "scene_path": scene_path,
        "object_type": type_str,
        "cache_directory": cache_dir,
        "parts": {}
    }

    # Load RGB Color Data
    color_data = {}
    color_file = f"{folder_path}/gpt4_query/final_rgb_colors.json"
    if os.path.exists(color_file):
        with open(color_file) as f:
            final_colors = json.load(f)
            color_data = {k: tuple(v) for k, v in final_colors.items() if isinstance(v, list)}

    print(f"\n{Colors.BOLD}{Colors.BLUE}[Step 9/9]{Colors.RESET} VLM Material Selection Pipeline...")
    print(f"  Scene: {Colors.BOLD}{scene_path}{Colors.RESET}")
    print(f"  Object Type: {Colors.BOLD}{type_str}{Colors.RESET}")
    print(f"  Parts to Process: {Colors.BOLD}{len(top_10_results)}{Colors.RESET}\n")

    for part_idx, (part_id, part_result_data) in enumerate(top_10_results.items(), 1):
        print(f"\n  {Colors.CYAN}📦 Part {part_idx}/{len(top_10_results)}:{Colors.RESET} {Colors.BOLD}{part_id}{Colors.RESET}")

        # Initialize part data
        part_data = {
            "part_id": part_id,
            "top_10_materials": [],
            "top_4_materials": [],
            "adapted_materials": [],
            "final_selected_material": None,
            "target_color": None,
            "image_paths": {
                "phase_1_top_10": [],
                "phase_2_top_4": [],
                "phase_3_adapted": [],
                "phase_4_final": None
            },
            "prompts": {}
        }

        # --- PHASE 1: TOP 10 MATERIALS ---
        phase1_start = time.time()
        print(f"    {Colors.BLUE}[Phase 1/4]{Colors.RESET} Top 10 Material Selection")
        print(f"      Analyzing {Colors.BOLD}{len(part_result_data.get('top_10', part_result_data.get('top_20', [])))}{Colors.RESET} candidates...")
        
        # Extract top 10 material IDs from Phase 1 results
        top_10_material_ids = part_result_data.get("top_10", part_result_data.get("top_20", []))
        if not top_10_material_ids:
            raise RuntimeError(f"CRITICAL: No material IDs found for {part_id} from Phase 1. Pipeline stopped.")

        part_data["top_10_materials"] = top_10_material_ids
        part_data["prompts"]["phase_1_top_10"] = part_result_data.get("prompt", "")
        
        # Save images for top 10 materials
        for mat_id in top_10_material_ids:
            albedo_path = load_material_albedo(mat_id)
            if albedo_path:
                img_path = save_albedo_image(mat_id, albedo_path, cache_dir, "phase_1_top_10", part_id)
                if img_path:
                    part_data["image_paths"]["phase_1_top_10"].append({
                        "material_id": mat_id,
                        "image_path": img_path
                    })
        phase1_elapsed = time.time() - phase1_start
        print(f"      {Colors.GREEN}✓{Colors.RESET} Phase 1 complete: Saved {len(part_data['image_paths']['phase_1_top_10'])} materials ({Colors.CYAN}{phase1_elapsed:.2f}s{Colors.RESET})")

        # --- PHASE 2: PATTERN ANALYSIS ---
        phase2_start = time.time()
        print(f"    {Colors.BLUE}[Phase 2/4]{Colors.RESET} Pattern Analysis")
        print(f"      Analyzing albedo patterns to reduce to top 4...")
        
        # Analyze albedo patterns to reduce to top 4 materials
        pattern_analysis_result = analyze_albedo_patterns(scene_path, part_id, top_10_material_ids, type_str)

        top_4_material_ids = pattern_analysis_result.get("top_4", [])

        # HARD STOP: If pattern analysis fails to find valid albedo textures
        if not top_4_material_ids:
            raise RuntimeError(f"CRITICAL: Phase 2 found 0 valid albedo textures for {part_id}. Pipeline stopped.")

        part_data["top_4_materials"] = top_4_material_ids
        part_data["prompts"]["phase_2_pattern_analysis"] = pattern_analysis_result.get("prompt", "")
        # Save images for top 4 materials
        for mat_id in top_4_material_ids:
            albedo_path = load_material_albedo(mat_id)
            if albedo_path:
                img_path = save_albedo_image(mat_id, albedo_path, cache_dir, "phase_2_top_4", part_id)
                if img_path:
                    part_data["image_paths"]["phase_2_top_4"].append({
                        "material_id": mat_id,
                        "image_path": img_path
                    })
        phase2_elapsed = time.time() - phase2_start
        print(f"      {Colors.GREEN}✓{Colors.RESET} Phase 2 complete: Selected {len(top_4_material_ids)} from {len(top_10_material_ids)} ({Colors.CYAN}{phase2_elapsed:.2f}s{Colors.RESET})")

        # --- PHASE 3: COLOR ADAPTATION ---
        phase3_start = time.time()
        print(f"    {Colors.BLUE}[Phase 3/4]{Colors.RESET} Color Adaptation")
        target_rgb_color = color_data.get(part_id, (128, 128, 128))
        part_data["target_color"] = target_rgb_color
        print(f"      Target Color: RGB{target_rgb_color}")
        print(f"      Adapting {len(top_4_material_ids)} materials to target color...")

        # Adapt the top 4 materials to match the target color
        color_adaptation_result = color_adapt_multiple_materials(scene_path, part_id, top_4_material_ids, target_rgb_color)
        adapted_material_list = color_adaptation_result.get("materials", [])

        # HARD STOP: If no materials survive the color adaptation
        if not adapted_material_list:
            raise RuntimeError(f"CRITICAL: Phase 3 produced 0 adapted materials for {part_id}. Pipeline stopped.")

        part_data["adapted_materials"] = adapted_material_list
        part_data["prompts"]["phase_3_color_adaptation"] = color_adaptation_result.get("prompt", "")
        
        # Save images for adapted materials
        for mat_info in adapted_material_list:
            mat_id = mat_info.get("material_id", "")
            adapted_path = mat_info.get("adapted_albedo", "")
            if adapted_path and os.path.exists(adapted_path):
                img_path = save_albedo_image(mat_id, adapted_path, cache_dir, "phase_3_adapted", part_id)
                if img_path:
                    part_data["image_paths"]["phase_3_adapted"].append({
                        "material_id": mat_id,
                        "image_path": img_path,
                        "original_material_id": mat_id
                    })
        phase3_elapsed = time.time() - phase3_start
        print(f"      {Colors.GREEN}✓{Colors.RESET} Phase 3 complete: Adapted {len(adapted_material_list)} materials ({Colors.CYAN}{phase3_elapsed:.2f}s{Colors.RESET})")

        # --- PHASE 4: FINAL SELECTION ---
        phase4_start = time.time()
        print(f"    {Colors.BLUE}[Phase 4/4]{Colors.RESET} Final Material Selection")
        print(f"      Selecting best from {len(adapted_material_list)} adapted materials...")
        
        # Select the best final material from the adapted candidates
        final_selection_result = select_final_material(scene_path, part_id, adapted_material_list, type_str)

        # HARD STOP: If final material selection fails
        if not final_selection_result or not final_selection_result.get("selected_material"):
            raise RuntimeError(f"CRITICAL: Phase 4 failed to select a final material for {part_id}. Pipeline stopped.")

        selected_material_id = final_selection_result["selected_material"]
        part_data["final_selected_material"] = selected_material_id
        part_data["prompts"]["phase_4_final_selection"] = final_selection_result.get("prompt", "")
        part_data["object_type"] = type_str
        part_data["scene_path"] = scene_path
        
        # Save final selected material image
        print(f"  💾 Saving final selected material image...")
        final_albedo_path = load_material_albedo(selected_material_id)
        if final_albedo_path:
            img_path = save_albedo_image(selected_material_id, final_albedo_path, cache_dir, "phase_4_final", part_id)
            if img_path:
                part_data["image_paths"]["phase_4_final"] = {
                    "material_id": selected_material_id,
                    "image_path": img_path
                }
        
        # Also save adapted version if available
        for mat_info in adapted_material_list:
            if mat_info.get("material_id") == selected_material_id:
                adapted_path = mat_info.get("adapted_albedo", "")
                if adapted_path and os.path.exists(adapted_path):
                    adapted_img_path = save_albedo_image(f"{selected_material_id}_adapted", adapted_path, cache_dir, "phase_4_final", part_id)
                    if adapted_img_path:
                        part_data["image_paths"]["phase_4_final_adapted"] = {
                            "material_id": selected_material_id,
                            "image_path": adapted_img_path
                        }
                break

        phase4_elapsed = time.time() - phase4_start
        part_total = phase1_elapsed + phase2_elapsed + phase3_elapsed + phase4_elapsed
        print(f"      {Colors.GREEN}✓{Colors.RESET} Phase 4 complete: Selected {Colors.BOLD}{selected_material_id}{Colors.RESET} ({Colors.CYAN}{phase4_elapsed:.2f}s{Colors.RESET})")
        print(f"    {Colors.GREEN}✓{Colors.RESET} Part {Colors.BOLD}{part_id}{Colors.RESET} complete ({Colors.CYAN}{part_total:.2f}s{Colors.RESET})")

        # Save part data to material selections
        material_selections["parts"][part_id] = part_data

    print(f"\n  {Colors.GREEN}✓{Colors.RESET} VLM Material Selection complete!")
    print(f"  Processed {Colors.BOLD}{len(material_selections['parts'])}{Colors.RESET} parts:")
    for part_id, part_data in material_selections["parts"].items():
        print(f"    • {Colors.BOLD}{part_id}{Colors.RESET}: {part_data['final_selected_material']}")
    print()

    # Save the material selections cache
    object_name = context.get("object_name", type_str)
    save_material_selection_cache(scene_path, object_name, material_selections)

def select_material(scene_path):
    """Main entry point for material selection pipeline"""
    pipeline_start = time.time()
    
    object_name = scene_path.split("/")[-1]
    
    # Print header
    print(f"\n{Colors.BOLD}{Colors.MAGENTA}╔═══════════════════════════════════════════════════════════════╗{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.MAGENTA}║{Colors.RESET} {Colors.BOLD}🎨 Material Selection Pipeline{Colors.RESET}                             {Colors.BOLD}{Colors.MAGENTA}║{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.MAGENTA}╚═══════════════════════════════════════════════════════════════╝{Colors.RESET}")
    print(f"{Colors.CYAN}Started at:{Colors.RESET} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{Colors.CYAN}Object:{Colors.RESET} {Colors.BOLD}{object_name}{Colors.RESET}\n")
    
    # Required Phase
    context = select_material_required_queries(scene_path)
    if not context: 
        print(f"{Colors.RED}❌{Colors.RESET} Material selection failed - context not initialized")
        return
    
    # VLM Phase
    run_vlm_material_pipeline(context)
    
    # Print footer
    pipeline_elapsed = time.time() - pipeline_start
    print(f"{Colors.BOLD}{Colors.GREEN}╔═══════════════════════════════════════════════════════════════╗{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GREEN}║{Colors.RESET} {Colors.BOLD}✓ Material Selection completed successfully!{Colors.RESET}              {Colors.BOLD}{Colors.GREEN}║{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GREEN}╚═══════════════════════════════════════════════════════════════╝{Colors.RESET}")
    print(f"{Colors.CYAN}Total time:{Colors.RESET} {Colors.BOLD}{pipeline_elapsed:.2f}s{Colors.RESET}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Material Selection Pipeline")
    parser.add_argument("--scene", type=str, required=True, help="Path to scene object folder")
    args = parser.parse_args()
    
    select_material(args.scene)