"""
Collects all material painting visualization data and saves to JSON.
Run this after material painting completes to gather all information.
"""

import json
import os
import glob
from pathlib import Path
from datetime import datetime

def collect_material_painting_log(scene_path):
    """
    Collect all material painting data for visualization
    
    Args:
        scene_path: Path to object folder (e.g., "output/mat_painting_stage/scene_1/Chair1")
    
    Returns:
        dict: Complete log data
    """
    object_name = os.path.basename(scene_path)
    object_name_clean = ''.join(filter(lambda x: not x.isdigit(), object_name))
    gpt_folder = f"{scene_path}/{object_name_clean}_gpt"
    gpt_query_folder = f"{gpt_folder}/gpt4_query"
    
    log = {
        "object_name": object_name,
        "object_name_clean": object_name_clean,
        "timestamp": datetime.now().isoformat(),
        "steps": {}
    }
    
    # Step 1: Auto-cropping
    cropped_dir = f"{scene_path}/cropped_image_all/{object_name_clean}"
    cropped_frames = []
    if os.path.exists(cropped_dir):
        for frame_dir in sorted(os.listdir(cropped_dir)):
            frame_path = os.path.join(cropped_dir, frame_dir)
            if os.path.isdir(frame_path):
                annotated = os.path.join(frame_path, "annotated_crops.jpg")
                crops = sorted(glob.glob(os.path.join(frame_path, "crop_*.jpg")))
                if annotated or crops:
                    cropped_frames.append({
                        "frame": frame_dir,
                        "annotated_image": annotated if os.path.exists(annotated) else None,
                        "cropped_images": crops[:10]  # Limit to first 10
                    })
    
    # Step 1: Auto-cropping (deprecated - no longer used in VLM pipeline)
    # Kept as placeholder for backward compatibility with UI
    log["steps"]["step_1_auto_cropping"] = {
        "status": "deprecated",
        "note": "Cropping is no longer used. Material selection now uses VLM pipeline."
    }
    
    # Step 2: Stitching - Only show segmentation summary
    segmentation_summary = f"{scene_path}/Onboarded/segmentation_summary.png"
    if os.path.exists(segmentation_summary):
        log["steps"]["step_2_stitching"] = {
            "status": "complete",
            "segmentation_summary": segmentation_summary
        }
    
    # Step 3: LLM Material Queries
    # Define paths needed for Step 3
    top_4_combine = f"{scene_path}/Onboarded/top_4_combine"
    captured_images_dir = f"{scene_path}/captured_images"
    top_9_labeled_dir = f"{scene_path}/Onboarded/top_9_labeled_images"
    stitched_image_path = os.path.join(captured_images_dir, "stitched_image.jpg")
    
    if os.path.exists(gpt_query_folder):
        llm_queries = {}
        
        # Check if new 4-step color query process was used
        holistic_colors_file = f"{gpt_query_folder}/holistic_colors.json"
        part_identification_file = f"{gpt_query_folder}/part_identification.json"
        rgb_voting_results_file = f"{gpt_query_folder}/rgb_voting_results.json"
        final_rgb_colors_file = f"{gpt_query_folder}/final_rgb_colors.json"
        validation_report_file = f"{gpt_query_folder}/validation_report.json"
        color_query_prompts_log_file = f"{gpt_query_folder}/color_query_prompts_log.json"
        
        uses_new_4step_process = (os.path.exists(holistic_colors_file) or 
                                  os.path.exists(part_identification_file) or 
                                  os.path.exists(rgb_voting_results_file))
        
        # Initialize color_query_4step dict
        color_query_4step = {}
        
        if uses_new_4step_process:
            # ========== NEW 4-STEP COLOR QUERY PROCESS ==========
            color_query_4step = {}
            
            # Step 1: Holistic Color Analysis
            if os.path.exists(holistic_colors_file):
                with open(holistic_colors_file) as f:
                    holistic_data = json.load(f)
                
                # Get input images (4 frame images)
                frame_images = []
                if os.path.exists(captured_images_dir):
                    preferred_frames = ['frame_12.jpg', 'frame_14.jpg', 'frame_15.jpg', 'frame_21.jpg']
                    for pref in preferred_frames:
                        pref_path = os.path.join(captured_images_dir, pref)
                        if os.path.exists(pref_path):
                            frame_images.append(pref_path)
                    
                    # If we don't have 4 preferred frames, add others
                    if len(frame_images) < 4:
                        for filename in sorted(os.listdir(captured_images_dir)):
                            if filename.startswith('frame_') and filename.endswith('.jpg'):
                                frame_path = os.path.join(captured_images_dir, filename)
                                if frame_path not in frame_images and len(frame_images) < 4:
                                    frame_images.append(frame_path)
                
                # Get prompt from log
                prompt = ""
                if os.path.exists(color_query_prompts_log_file):
                    with open(color_query_prompts_log_file) as f:
                        prompts_log = json.load(f)
                        if 'step1_holistic' in prompts_log:
                            prompt = prompts_log['step1_holistic'].get('prompt', '')
                            input_images_log = prompts_log['step1_holistic'].get('input_images', [])
                            if input_images_log:
                                frame_images = input_images_log
                
                color_query_4step["step1_holistic_color_analysis"] = {
                    "method": "multi_image_query",
                    "input_images": frame_images,
                    "prompt": prompt,
                    "response": holistic_data,
                    "response_file": holistic_colors_file,
                    "object_type": holistic_data.get("object_type", ""),
                    "main_colors": holistic_data.get("main_colors", {}),
                    "color_details": holistic_data.get("color_details", {}),
                    "overall_theme": holistic_data.get("overall_theme", "")
                }
            
            # Step 2: Part Identification Enhanced
            if os.path.exists(part_identification_file):
                with open(part_identification_file) as f:
                    part_identification_data = json.load(f)
                
                # Get prompts from log
                prompts_log = {}
                if os.path.exists(color_query_prompts_log_file):
                    with open(color_query_prompts_log_file) as f:
                        prompts_log = json.load(f)
                
                part_identification_details = {}
                for part_id, part_data in part_identification_data.items():
                    # Get top 4 labeled images for this part
                    part_folder = os.path.join(top_9_labeled_dir, part_id)
                    labeled_images = []
                    if os.path.exists(part_folder):
                        for filename in sorted(os.listdir(part_folder)):
                            if filename.endswith('_labeled.png'):
                                labeled_images.append(os.path.join(part_folder, filename))
                        labeled_images = labeled_images[:4]  # Top 4
                    
                    part_info = {
                        "part_id": part_id,
                        "part_name": part_data.get("part_name", ""),
                        "description": part_data.get("description", ""),
                        "typical_colors": part_data.get("typical_colors", []),
                        "confidence": part_data.get("confidence", ""),
                        "input_images": labeled_images,
                        "response": part_data
                    }
                    
                    # Get prompt if available (from log)
                    if 'step2_identification' in prompts_log:
                        prompt_template = prompts_log['step2_identification'].get('prompt_template', '')
                        part_info["prompt_template"] = prompt_template
                    
                    part_identification_details[part_id] = part_info
                
                color_query_4step["step2_part_identification"] = {
                    "method": "multi_image_query_per_part",
                    "parts": part_identification_details,
                    "response_file": part_identification_file
                }
            
            # Step 3: Per-Part Color Query with Voting
            if os.path.exists(rgb_voting_results_file):
                with open(rgb_voting_results_file) as f:
                    voting_results_data = json.load(f)
                
                # Get prompts from log
                prompts_log = {}
                if os.path.exists(color_query_prompts_log_file):
                    with open(color_query_prompts_log_file) as f:
                        prompts_log = json.load(f)
                
                voting_details = {}
                for part_id, voting_data in voting_results_data.items():
                    # Get the 4 labeled images used for voting
                    part_folder = os.path.join(top_9_labeled_dir, part_id)
                    labeled_images = []
                    if os.path.exists(part_folder):
                        for filename in sorted(os.listdir(part_folder)):
                            if filename.endswith('_labeled.png'):
                                labeled_images.append(os.path.join(part_folder, filename))
                        labeled_images = labeled_images[:4]  # Top 4 used for voting
                    
                    # Get prompts for each query (4 different prompts)
                    prompt_templates = []
                    if 'step3_voting' in prompts_log:
                        prompt_templates = prompts_log['step3_voting'].get('prompt_templates', [])
                    
                    # Construct query details for each of the 4 votes
                    queries = []
                    rgb_votes = voting_data.get('rgb_votes', [])
                    for query_idx, rgb_vote in enumerate(rgb_votes[:4]):
                        query_info = {
                            "query_number": query_idx + 1,
                            "input_images": [
                                stitched_image_path if os.path.exists(stitched_image_path) else None,
                                labeled_images[query_idx] if query_idx < len(labeled_images) else None
                            ],
                            "prompt": prompt_templates[query_idx] if query_idx < len(prompt_templates) else "",
                            "rgb_response": rgb_vote
                        }
                        queries.append(query_info)
                    
                    voting_details[part_id] = {
                        "part_id": part_id,
                        "part_name": voting_data.get("part_name", ""),
                        "queries": queries,  # All 4 queries with images and prompts
                        "rgb_final": voting_data.get("rgb_final", []),
                        "rgb_votes": voting_data.get("rgb_votes", []),
                        "consensus_score": voting_data.get("consensus_score", 0.0),
                        "confidence": voting_data.get("confidence", ""),
                        "method": voting_data.get("method", ""),
                        "estimation_method": voting_data.get("estimation_method", ""),
                        "visibility": voting_data.get("visibility", "")
                    }
                
                color_query_4step["step3_color_voting"] = {
                    "method": "4_query_voting_per_part",
                    "parts": voting_details,
                    "response_file": rgb_voting_results_file
                }
            
            # Step 4: Validation & Refinement
            if os.path.exists(final_rgb_colors_file) and os.path.exists(validation_report_file):
                with open(final_rgb_colors_file) as f:
                    final_colors_data = json.load(f)
                
                with open(validation_report_file) as f:
                    validation_data = json.load(f)
                
                # Get re-query details if any
                re_query_details = {}
                validation_results = validation_data.get("validation_results", {})
                for part_id, val_result in validation_results.items():
                    if val_result.get("flag_level") == 2 and val_result.get("iteration", 0) > 0:
                        # This part was re-queried
                        re_query_info = {
                            "part_id": part_id,
                            "iteration": val_result.get("iteration", 0),
                            "step1_color": val_result.get("step1_color", []),
                            "step3_color_before": val_result.get("step3_color", []),
                            "step3_color_after": final_colors_data.get(part_id, []),
                            "rgb_distance_before": val_result.get("rgb_distance", 0),
                            "status": val_result.get("status", "")
                        }
                        re_query_details[part_id] = re_query_info
                
                color_query_4step["step4_validation_refinement"] = {
                    "method": "consistency_check_and_refinement",
                    "final_colors": final_colors_data,
                    "validation_results": validation_results,
                    "missing_colors": validation_data.get("missing_colors", []),
                    "segmentation_issues": validation_data.get("segmentation_issues", []),
                    "overall_confidence": validation_data.get("overall_confidence", ""),
                    "re_query_details": re_query_details,
                    "response_file": final_rgb_colors_file,
                    "validation_file": validation_report_file
                }
            
            if color_query_4step:
                llm_queries["color_query_4step"] = {
                    "method": "4-step_color_query_with_voting",
                    "steps": color_query_4step
                }
        
        # ========== LEGACY: Old single-pass color query (for backward compatibility) ==========
        # 3.1 Object Structure
        shape_file = f"{gpt_query_folder}/shape_result.json"
        prompt_log_file = f"{gpt_query_folder}/object_structure_prompt.json"
        if os.path.exists(shape_file) and "object_structure" not in llm_queries:
            with open(shape_file) as f:
                shape_data = json.load(f)
            
            # Load prompt if available
            prompt = ""
            if os.path.exists(prompt_log_file):
                with open(prompt_log_file) as f:
                    prompt_data = json.load(f)
                    prompt = prompt_data.get("prompt", "")
            
            # Get first entry (usually only one)
            for img_path, response in shape_data.items():
                llm_queries["object_structure"] = {
                    "input_image": img_path,
                    "prompt": prompt if prompt else "Object structure analysis prompt",
                    "response": response.get("choices", [{}])[0].get("message", {}).get("content", ""),
                    "response_file": shape_file
                }
                break
        
        # 3.2 Part Descriptions (from segmentation.json)
        segmentation_file = f"{gpt_query_folder}/segmentation.json"
        if os.path.exists(segmentation_file) and "part_descriptions" not in llm_queries:
            with open(segmentation_file) as f:
                segmentation_data = json.load(f)
            
            # Get input images - these queries use top_4_combine images
            top_4_combine_dir = f"{scene_path}/Onboarded/top_4_combine"
            input_images = {}
            part_descriptions = {}
            if os.path.exists(top_4_combine_dir):
                for part_name in segmentation_data.keys():
                    part_img = os.path.join(top_4_combine_dir, f"{part_name}.png")
                    if os.path.exists(part_img):
                        input_images[part_name] = part_img
                    
                    # Extract response
                    response_data = segmentation_data[part_name]
                    part_descriptions[part_name] = {
                        "input_image": input_images.get(part_name),
                        "response": response_data.get("choices", [{}])[0].get("message", {}).get("content", ""),
                        "response_file": segmentation_file
                    }
            
            if part_descriptions:
                llm_queries["part_descriptions"] = {
                    "method": "per_part_query",
                    "input_images": input_images,
                    "parts": part_descriptions,
                    "response_file": segmentation_file,
                    "description": "Part descriptions from segmentation analysis"
                }
        
        # 3.2b Part Descriptions (from group_segment.json - alternative format)
        group_file = f"{gpt_query_folder}/group_segment.json"
        if os.path.exists(group_file) and "part_descriptions_grouped" not in llm_queries:
            with open(group_file) as f:
                group_data = json.load(f)
            llm_queries["part_descriptions_grouped"] = {
                "method": "grouped_segments",
                "response": group_data,
                "response_file": group_file,
                "description": "Grouped part descriptions"
            }
        
        # 3.3 Part Colors (per part) - Load from prompt log if available (LEGACY)
        # Only collect legacy part_colors if new 4-step process was NOT used
        rgb_file = f"{gpt_query_folder}/rgb_only.json"
        prompt_log_file = f"{gpt_query_folder}/color_extraction_prompts.json"
        
        if os.path.exists(rgb_file) and "part_colors" not in llm_queries and not uses_new_4step_process:
            with open(rgb_file) as f:
                rgb_data = json.load(f)
            
            # Load prompts if available
            prompt_log = {}
            if os.path.exists(prompt_log_file):
                with open(prompt_log_file) as f:
                    prompt_log = json.load(f)
            
            part_colors = {}
            for part_name, rgb_response in rgb_data.items():
                stitched_part_img = os.path.join(top_4_combine, f"{part_name}.png")
                part_info = {
                    "input_image": stitched_part_img if os.path.exists(stitched_part_img) else None,
                    "response": rgb_response,
                    "response_file": rgb_file
                }
                
                # Add prompt if available
                if part_name in prompt_log:
                    part_info["prompt"] = prompt_log[part_name].get("prompt", "")
                    part_info["part_description"] = prompt_log[part_name].get("part_description", "")
                
                part_colors[part_name] = part_info
            
            llm_queries["part_colors"] = {
                "method": "legacy_single_pass",
                "parts": part_colors,
                "all_colors_file": rgb_file
            }
        
        # 3.4 Crop Selection (deprecated - no longer used in VLM pipeline)
        # This section is kept empty for backward compatibility with old data
        
        # 3.5 Material Types (First Pass)
        overall_file = f"{gpt_query_folder}/overall_query.json"
        if os.path.exists(overall_file):
            with open(overall_file) as f:
                overall_data = json.load(f)
            
            # Get input images - these queries use stitched part images
            input_images = {}
            if overall_data:
                for part_name in overall_data.keys():
                    stitched_part_img = os.path.join(top_4_combine, f"{part_name}.png")
                    if os.path.exists(stitched_part_img):
                        input_images[part_name] = stitched_part_img
            
            llm_queries["material_types_pass1"] = {
                "method": "per_part_query",
                "input_images": input_images,
                "response": overall_data,
                "response_file": overall_file,
                "description": "First pass material type determination for each part"
            }
        
        # 3.6 Material Types (Second Pass)
        sub_cat_file = f"{gpt_query_folder}/sub_cat.json"
        if os.path.exists(sub_cat_file):
            with open(sub_cat_file) as f:
                sub_cat_data = json.load(f)
            
            # Get input images (stitched part images) for each part
            input_images = {}
            if sub_cat_data:
                for part_name in sub_cat_data.keys():
                    stitched_part_img = os.path.join(top_4_combine, f"{part_name}.png")
                if os.path.exists(stitched_part_img):
                        input_images[part_name] = stitched_part_img
            
            llm_queries["material_types_pass2"] = {
                "method": "per_part_query",
                "input_images": input_images,
                "response": sub_cat_data,
                "response_file": sub_cat_file,
                "description": "Second pass material type refinement for each part"
            }
        
        # Add summary of which methods were used
        if llm_queries:
            methods_used = []
            if "color_query_4step" in llm_queries:
                methods_used.append("4-step_color_query_with_voting")
            if "part_colors" in llm_queries:
                methods_used.append("legacy_single_pass_color_query")
            
            log["steps"]["step_3_llm_material_retrieval"] = {
                "status": "complete",
                "method": "LLM (Qwen3-VL)",
                "methods_used": methods_used,
                "uses_new_4step_process": uses_new_4step_process,
                "queries": llm_queries
            }
    
    # Step 3.5: VLM Material Selection Pipeline
    vlm_conversations_dir = f"{gpt_folder}/vlm_conversations"
    top_20_file = f"{gpt_query_folder}/top_20_materials.json"
    top_4_pattern_file = f"{gpt_query_folder}/top_4_pattern_matched.json"
    final_selection_file = f"{gpt_query_folder}/final_material_selection.json"
    color_adapted_dir = f"{scene_path}/color_adapted_materials"
    
    vlm_material_selection = {}
    
    if os.path.exists(vlm_conversations_dir):
        from litereality.LR_mat_painting.utils.vlm_logger import get_all_conversations
        
        # Load all phase results (check both top_10 and top_20 files for backward compatibility)
        top_20_data = {}
        top_10_file = f"{gpt_query_folder}/top_10_materials.json"
        if os.path.exists(top_10_file):
            with open(top_10_file) as f:
                top_20_data = json.load(f)
        elif os.path.exists(top_20_file):
            with open(top_20_file) as f:
                top_20_data = json.load(f)
        
        top_4_pattern_data = {}
        if os.path.exists(top_4_pattern_file):
            with open(top_4_pattern_file) as f:
                top_4_pattern_data = json.load(f)
        
        final_selection_data = {}
        if os.path.exists(final_selection_file):
            with open(final_selection_file) as f:
                final_selection_data = json.load(f)
        
        # Get all conversations and organize by part/phase
        all_conversations = get_all_conversations(scene_path, object_name_clean)
        
        # Create conversation index by part_id and phase
        conv_index = {}
        for conv in all_conversations:
            part_id = conv.get("part_id")
            phase = conv.get("phase", "")
            if part_id and phase:
                if part_id not in conv_index:
                    conv_index[part_id] = {}
                conv_index[part_id][phase] = conv
        
        # Organize by part and phase
        for part_id in set(list(top_20_data.keys()) + list(top_4_pattern_data.keys()) + list(final_selection_data.keys())):
            part_data = {
                "phase_1_top_20": {},
                "phase_2_pattern_analysis": {},
                "phase_3_color_adaptation": {},
                "phase_4_final_selection": {}
            }
            
            # Phase 1 data (support both top_10 and top_20 keys)
            if part_id in top_20_data:
                # Get materials from top_10 (new) or top_20 (old) key
                candidates = top_20_data[part_id].get("top_10", top_20_data[part_id].get("top_20", []))
                part_data["phase_1_top_20"] = {
                    "candidates": candidates,
                    "scores": top_20_data[part_id].get("scores", {}),
                    "reasoning": top_20_data[part_id].get("reasoning", "")
                }
                # Add conversation log path
                phase1_conv = conv_index.get(part_id, {}).get("phase_1_text_visual_selection")
                if phase1_conv:
                    vlm_query_folder = f"{gpt_folder}/vlm_conversations"
                    conv_file = f"{vlm_query_folder}/phase_1_text_visual_selection_part_{part_id}.json"
                    if os.path.exists(conv_file):
                        part_data["phase_1_top_20"]["vlm_conversation_log"] = os.path.relpath(conv_file, scene_path)
            
            # Phase 2 data
            if part_id in top_4_pattern_data:
                part_data["phase_2_pattern_analysis"] = {
                    "top_4": top_4_pattern_data[part_id].get("top_4", []),
                    "scores": top_4_pattern_data[part_id].get("scores", {}),
                    "reasoning": top_4_pattern_data[part_id].get("reasoning", "")
                }
                # Add conversation log path
                phase2_conv = conv_index.get(part_id, {}).get("phase_2_pattern_analysis")
                if phase2_conv:
                    vlm_query_folder = f"{gpt_folder}/vlm_conversations"
                    conv_file = f"{vlm_query_folder}/phase_2_pattern_analysis_part_{part_id}.json"
                    if os.path.exists(conv_file):
                        part_data["phase_2_pattern_analysis"]["vlm_conversation_log"] = os.path.relpath(conv_file, scene_path)
            
            # Phase 3 data
            part_color_adapted_dir = os.path.join(color_adapted_dir, part_id)
            if os.path.exists(part_color_adapted_dir):
                adaptation_results_file = os.path.join(part_color_adapted_dir, "color_adaptation_results.json")
                if os.path.exists(adaptation_results_file):
                    with open(adaptation_results_file) as f:
                        adaptation_data = json.load(f)
                        part_data["phase_3_color_adaptation"] = {
                            "target_rgb": adaptation_data.get("target_rgb", []),
                            "materials": adaptation_data.get("materials", [])
                        }
            
            # Phase 4 data
            if part_id in final_selection_data:
                part_data["phase_4_final_selection"] = final_selection_data[part_id]
                # Add conversation log path
                phase4_conv = conv_index.get(part_id, {}).get("phase_4_final_selection")
                if phase4_conv:
                    vlm_query_folder = f"{gpt_folder}/vlm_conversations"
                    conv_file = f"{vlm_query_folder}/phase_4_final_selection_part_{part_id}.json"
                    if os.path.exists(conv_file):
                        part_data["phase_4_final_selection"]["vlm_conversation_log"] = os.path.relpath(conv_file, scene_path)
            
            vlm_material_selection[part_id] = part_data
        
        if vlm_material_selection:
            log["steps"]["step_3_5_vlm_material_selection"] = {
                "status": "complete",
                "method": "VLM_pipeline",
                "parts": vlm_material_selection
            }
    
    # Step 4: Visual Material Retrieval (Deprecated - CLIP-based, replaced by VLM pipeline)
    selected_material_dir = f"{scene_path}/selected_material/{object_name_clean}"
    visual_retrieval_log_file = f"{scene_path}/visual_retrieval_log.json"
    
    if os.path.exists(visual_retrieval_log_file):
        # Load from dedicated log file if available
        with open(visual_retrieval_log_file) as f:
            visual_retrieval_data = json.load(f)
        
        # Convert selected_material_path strings to selected_material objects with basecolor
        if "parts" in visual_retrieval_data:
            for part_name, part_data in visual_retrieval_data["parts"].items():
                if "selected_material_path" in part_data and "selected_material" not in part_data:
                    # Try to find the material files
                    material_path = part_data["selected_material_path"]
                    
                    # Check multiple possible locations (try backup first since originals may be deleted)
                    possible_paths = [
                        f"{scene_path}/selected_material_backup_1000/{object_name_clean}_gpt/{part_name}",  # Backup location (resized)
                        material_path if material_path else None,  # Original path from log (only if not None)
                        f"{scene_path}/selected_material/{object_name_clean}_gpt/{part_name}",  # Standard location
                    ]
                    
                    # Filter out None values
                    possible_paths = [p for p in possible_paths if p is not None]
                    
                    basecolor_path = None
                    for path in possible_paths:
                        if path is None:
                            continue
                        basecolor_file = os.path.join(path, "basecolor.png")
                        if os.path.exists(basecolor_file):
                            basecolor_path = basecolor_file
                            break
                    
                    if basecolor_path:
                        part_data["selected_material"] = {
                            "path": os.path.dirname(basecolor_path),
                            "basecolor": basecolor_path
                        }
        
        log["steps"]["step_4_visual_material_retrieval"] = {
            "status": "complete",
            "method": "CLIP",
            **visual_retrieval_data
        }
    elif os.path.exists(selected_material_dir):
        # Fallback: reconstruct from directory structure
        parts_data = {}
        for part_dir in os.listdir(selected_material_dir):
            part_path = os.path.join(selected_material_dir, part_dir)
            if os.path.isdir(part_path):
                basecolor = os.path.join(part_path, "basecolor.png")
                if os.path.exists(basecolor):
                    parts_data[part_dir] = {
                        "selected_material": {
                            "path": part_path,
                            "basecolor": basecolor,
                            "normal": os.path.join(part_path, "normal.png") if os.path.exists(os.path.join(part_path, "normal.png")) else None,
                            "roughness": os.path.join(part_path, "roughness.png") if os.path.exists(os.path.join(part_path, "roughness.png")) else None
                        }
                    }
        
        # Note: Crop-related data collection removed (deprecated)
        
        if parts_data:
            log["steps"]["step_4_visual_material_retrieval"] = {
                "status": "deprecated",
                "method": "CLIP",
                "note": "CLIP-based visual retrieval is deprecated. New pipeline uses VLM material selection.",
                "parts": parts_data
            }
    
    # Step 5: Material Refinement
    refined_material_dir = f"{scene_path}/selected_material_OT_with_adaptation/{object_name_clean}"
    refined_material_dir_1000 = f"{scene_path}/selected_material_OT_with_adaptation_1000/{object_name_clean}_gpt"
    before_material_backup_dir = f"{scene_path}/selected_material_backup_1000/{object_name_clean}_gpt"
    refinement_log_file = f"{scene_path}/material_refinement_log.json"
    
    if os.path.exists(refinement_log_file):
        # Load from dedicated log file if available
        with open(refinement_log_file) as f:
            refinement_data = json.load(f)
        
        # Fix paths in refinement_data to point to actual file locations
        if "parts" in refinement_data:
            for part_name, part_data in refinement_data["parts"].items():
                # Fix "before" material path - check backup location first
                if "before_material" in part_data and "basecolor" in part_data["before_material"]:
                    before_path = part_data["before_material"]["basecolor"]
                    # Check if file exists, if not try backup location
                    if not os.path.exists(before_path):
                        backup_path = os.path.join(before_material_backup_dir, part_name, "basecolor.png")
                        if os.path.exists(backup_path):
                            part_data["before_material"]["basecolor"] = backup_path
                
                # Fix "after" material path - check _1000 location first
                if "after_material" in part_data and "basecolor" in part_data["after_material"]:
                    after_path = part_data["after_material"]["basecolor"]
                    # Check if file exists, if not try _1000 location
                    if not os.path.exists(after_path):
                        after_path_1000 = os.path.join(refined_material_dir_1000, part_name, "basecolor.png")
                        if os.path.exists(after_path_1000):
                            part_data["after_material"]["basecolor"] = after_path_1000
        
        log["steps"]["step_5_material_refinement"] = {
            "status": "complete",
            **refinement_data
        }
    elif os.path.exists(refined_material_dir):
        # Fallback: reconstruct from directory structure
        refinement_data = {
            "method": "color_adaptation",
            "parts": {}
        }
        rgb_file = f"{gpt_query_folder}/rgb_only.json"
        rgb_data = {}
        if os.path.exists(rgb_file):
            with open(rgb_file) as f:
                rgb_data = json.load(f)
        
        for part_dir in os.listdir(refined_material_dir):
            part_path = os.path.join(refined_material_dir, part_dir)
            if os.path.isdir(part_path):
                before_basecolor = os.path.join(selected_material_dir, part_dir, "basecolor.png")
                after_basecolor = os.path.join(part_path, "basecolor.png")
                
                if os.path.exists(after_basecolor):
                    refinement_data["parts"][part_dir] = {
                        "target_rgb": rgb_data.get(part_dir, None),
                        "before_material": {
                            "basecolor": before_basecolor if os.path.exists(before_basecolor) else None
                        },
                        "after_material": {
                            "basecolor": after_basecolor
                        }
                    }
        
        if refinement_data["parts"]:
            log["steps"]["step_5_material_refinement"] = {
                "status": "complete",
                **refinement_data
            }
    
    # Step 6: Final Render
    final_render = f"{scene_path}/A-ReTextured/OT_refined_with_adaptation/render_image_0.png"
    reference_image = f"{scene_path}/captured_images/stitched_image.jpg"
    if os.path.exists(final_render):
        log["steps"]["step_6_final_render"] = {
            "status": "complete",
            "rendered_image": final_render,
            "reference_image": reference_image if os.path.exists(reference_image) else None
        }
    
    return log

def save_material_painting_log(scene_path, output_file=None):
    """Collect and save material painting log"""
    log = collect_material_painting_log(scene_path)
    
    if output_file is None:
        output_file = f"{scene_path}/material_painting_log.json"
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(log, f, indent=2)
    
    print(f"✅ Material painting log saved to: {output_file}")
    return log

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True, help="Path to object folder")
    parser.add_argument("--output", default=None, help="Output JSON file path")
    args = parser.parse_args()
    
    save_material_painting_log(args.scene, args.output)

