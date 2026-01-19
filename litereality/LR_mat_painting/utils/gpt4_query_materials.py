"""
Material Query Module for Material Painting Pipeline

This module provides functions for querying VLM (Vision-Language Models) to:
- Identify object shapes and materials
- Extract color information from images
- Select appropriate materials from the database
- Validate and refine color selections

The module uses Qwen3-VL for all VLM queries and supports a multi-step
color extraction process with voting and holistic validation.
"""

# Standard library imports
import base64
import json
import os
import re
import sys
import time

# Third-party imports
from PIL import Image
from tqdm import tqdm
import requests

# Local imports
from .qwen_model_manager import ensure_model_loaded
from .qwen_query_materials import process_image_qwen, process_image_multi_qwen, QWEN_AVAILABLE

# Import prompts from centralized prompts module
from litereality.LR_mat_painting.prompts import (
    # Base prompts
    ADD_PROMPT,
    ADD_PROMPT_DOUBLE,
    PROMPT_OVERALL,
    INTRODUCTION_PROMPT,
    SEGMENTATION_PROMPT,
    SEGMENT_GROUP_PROMPT,
    COLOR_INFORMATION_EXTRACTION,
    PROMPT_SHAPE,
    QUERY_CROPPING,
    # Material prompts
    VALID_MATERIALS,
    MATERIAL_MAPPING,
    PROMPT_REFINE,
    PROMPT_REFINE_NEW,
    PROMPT_REFINE_SUB,
    PROMPT_REFINE_SUB_OVERALL,
    check_material_valid,
    # VLM phase prompts
    get_holistic_color_prompt,
    get_part_identification_prompt,
    get_color_voting_prompt,
    get_top_10_selection_prompt,
    get_semantic_mapping_prompt,
    get_color_verification_prompt,
)

# Backward compatibility aliases for module-level prompt constants
add_prompt = ADD_PROMPT
add_prompt_double = ADD_PROMPT_DOUBLE
prompt_overall = PROMPT_OVERALL
introduction_prompt = INTRODUCTION_PROMPT
segmentation_prompt = SEGMENTATION_PROMPT
segment_group_prompt = SEGMENT_GROUP_PROMPT
color_information_extraction = COLOR_INFORMATION_EXTRACTION
prompt_shape = PROMPT_SHAPE
query_cropping = QUERY_CROPPING
valid_mat = VALID_MATERIALS
prompt_refine = PROMPT_REFINE
prompt_refine_new = PROMPT_REFINE_NEW
prompt_refine_sub = PROMPT_REFINE_SUB
prompt_refine_sub_overall = PROMPT_REFINE_SUB_OVERALL


def _get_response_json(response):
    """
    Helper function to extract JSON dict from response.
    Handles both OpenAI client response objects and dict responses from Qwen.
    """
    if isinstance(response, dict):
        return response
    elif hasattr(response, 'json'):
        # requests Response object
        return response.json()
    elif hasattr(response, 'model_dump'):
        # OpenAI client response object
        return response.model_dump()
    else:
        # Try to convert to dict
        return dict(response) if hasattr(response, '__dict__') else response


def check_mat_valid(mat, count):
    """
    Validate and normalize a material name.

    Uses the centralized check_material_valid function from prompts module.

    Args:
        mat: Material name to validate
        count: Current retry count (fallback to 'Misc' after 3)

    Returns:
        Valid material name, mapped name, 'Misc' (after 3 retries), or None
    """
    return check_material_valid(mat, count)


def process_image(image_path, api_key, prompt):
    """
    Process image with Qwen3-VL (replaces OpenAI API call).
    api_key parameter kept for backward compatibility but not used.
    
    Returns a dict that mimics OpenAI response format for compatibility.
    """
    if not QWEN_AVAILABLE:
        raise RuntimeError("Qwen3-VL not available. Please ensure qwan.py is accessible.")
    
    # Ensure model is loaded
    ensure_model_loaded()
    
    # Use Qwen inference
    return process_image_qwen(image_path, prompt, max_new_tokens=300, max_image_size=800)

def query_shape(folder_path, img_pth, api_key, type_str=None):
    """Query overall information(e.g., shape, type) of 3D object.(help to query material)"""
    cache1_pth = f"{folder_path}/gpt4_query/shape_result.json"
    os.makedirs(os.path.split(cache1_pth)[0], exist_ok=True)
    if not os.path.exists(cache1_pth):
        json.dump({}, open(cache1_pth, 'w'), indent=4)
    data = json.load(open(cache1_pth))
    response, count, success = '', 0, False

    intro_prompt = add_prompt.format(type_str, type_str)
    if img_pth not in data.keys():
        while count < 5:
            try:
                new_prompt_shape = intro_prompt  + prompt_shape
                print("--" * 20)
                print("new_prompt_shape", new_prompt_shape)
                print("--" * 20)

                response = process_image(img_pth, api_key, new_prompt_shape)
                if 'error' in _get_response_json(response).keys():
                    print("waiting for rate limit...")
                    time.sleep(5)
                    continue
                print(f"success: query {img_pth}")
                success = True
                break
            except:
                count += 1
                print(f"error: query {img_pth}")
                time.sleep(5)
                continue
        if success:        
            data[img_pth] = _get_response_json(response)
            json.dump(data, open(cache1_pth, 'w'), indent=4)
            
            # Save prompt to log file
            prompt_log_file = f"{folder_path}/gpt4_query/object_structure_prompt.json"
            prompt_log = {
                "prompt": new_prompt_shape,
                "input_image": img_pth,
                "type_str": type_str
            }
            with open(prompt_log_file, 'w') as f:
                json.dump(prompt_log, f, indent=4)
    if count == 5: return None
    if data[img_pth] != "":
        shape_info = data[img_pth]["choices"][0]["message"]["content"]
        print(shape_info)
        return shape_info
    else:
        return None

def query_part_merged(folder_path, img_pth, api_key, type_str=None):

    previous_query = folder_path +  "/gpt4_query"
    segmentation_infor = os.path.join(previous_query, "segmentation.json")

    # Load segmentation data
    with open(segmentation_infor, 'r') as file:
        data = json.load(file)

    # Extract descriptions
    descriptions = {
        key: value["choices"][0]["message"]["content"] for key, value in data.items()
    }

    summary_path = img_pth + "/../" + "mask_merging.jpg"
    index_dict_path = img_pth + "/../" + "index_dict.json"

    # load index_dict

    with open(index_dict_path, 'r') as file:
        index_dict = json.load(file)

    #change the description key to index
    descriptions = {index_dict[key]: value for key, value in descriptions.items()}
    # sorted the descriptions by its key 
    descriptions = dict(sorted(descriptions.items()))
    prompt = segment_group_prompt.format(descriptions)

    group_seg_pth = f"{folder_path}/gpt4_query/group_segment.json"
    os.makedirs(os.path.split(group_seg_pth)[0], exist_ok=True)
    if not os.path.exists(group_seg_pth):
        json.dump({}, open(group_seg_pth, 'w'), indent=4)
    data = json.load(open(group_seg_pth))
    response, count, success = '', 0, False
    intro_prompt = prompt
    img_pth = summary_path
    if img_pth not in data.keys():
        while count < 5:
            try:
                new_prompt_shape = intro_prompt
                response = process_image(img_pth, api_key, new_prompt_shape)
                if 'error' in _get_response_json(response).keys():
                    print("waiting for rate limit...")
                    time.sleep(5)
                    continue
                print(f"success: query {img_pth}")
                success = True
                break
            except:
                count += 1
                print(f"error: query {img_pth}")
                time.sleep(5)
                continue
        if success:        
            data[img_pth] = _get_response_json(response)
            json.dump(data, open(group_seg_pth, 'w'), indent=4)
    if count == 5: return None
    if data[img_pth] != "":
        shape_info = data[img_pth]["choices"][0]["message"]["content"]
        print("output_information", shape_info)
        return shape_info
    else:
        return None


# ============================================================================
# Helper Functions for 4-Step Color Query Strategy
# ============================================================================

def parse_rgb_from_response(response_text):
    """
    Extract RGB values from LLM response text.
    Handles multiple formats: [R, G, B], (R, G, B), R, G, B, JSON arrays, etc.
    
    Args:
        response_text: Text response from LLM
    
    Returns:
        tuple: (R, G, B) as integers, or None if parsing fails
    """
    if not response_text:
        return None
    
    # Clean the response text
    text = response_text.strip()
    
    # Try to extract RGB using regex patterns
    patterns = [
        r'\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]',  # [R, G, B]
        r'\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\)',  # (R, G, B)
        r'(\d+)\s*,\s*(\d+)\s*,\s*(\d+)',      # R, G, B
        r'RGB[:\s]*\[?(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]?',  # RGB: [R, G, B]
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                r, g, b = int(match.group(1)), int(match.group(2)), int(match.group(3))
                # Validate RGB range
                if 0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255:
                    return (r, g, b)
            except (ValueError, IndexError):
                continue
    
    # Try JSON parsing if response looks like JSON
    try:
        # Try to find JSON array in the text
        json_match = re.search(r'\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]', text)
        if json_match:
            r, g, b = int(json_match.group(1)), int(json_match.group(2)), int(json_match.group(3))
            if 0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255:
                return (r, g, b)
    except:
        pass
    
    return None


def calculate_rgb_distance(rgb1, rgb2):
    """
    Calculate Euclidean distance between two RGB colors.
    
    Args:
        rgb1: Tuple or list of (R, G, B)
        rgb2: Tuple or list of (R, G, B)
    
    Returns:
        float: Euclidean distance in RGB space
    """
    if rgb1 is None or rgb2 is None:
        return float('inf')
    
    r1, g1, b1 = rgb1
    r2, g2, b2 = rgb2
    
    return ((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2) ** 0.5


def find_best_holistic_match_for_votes(vote_rgbs, holistic_colors):
    """
    When votes are moderately different, select the holistic color that best matches
    the majority of votes, rather than averaging.

    Args:
        vote_rgbs: List of RGB values from votes
        holistic_colors: Dict of holistic colors from Step 1

    Returns:
        dict with 'rgb', 'part_name', 'match_score' or None
    """
    if not holistic_colors or not isinstance(holistic_colors, dict):
        return None

    step1_color_details = holistic_colors.get('color_details', {})
    if not step1_color_details:
        return None

    # Count how many votes match each holistic color (within 25 RGB distance)
    holistic_matches = {}
    for part_name, color_data in step1_color_details.items():
        holistic_rgb = color_data.get('rgb_estimate')
        if not holistic_rgb or not isinstance(holistic_rgb, list):
            continue

        match_count = 0
        total_distance = 0

        for vote_rgb in vote_rgbs:
            distance = calculate_rgb_distance(holistic_rgb, vote_rgb)
            if distance < 25:  # Consider it a match if close enough
                match_count += 1
                total_distance += distance

        if match_count > 0:
            avg_distance = total_distance / match_count
            holistic_matches[part_name] = {
                'rgb': holistic_rgb,
                'match_count': match_count,
                'avg_distance': avg_distance,
                'score': match_count / len(vote_rgbs)  # Percentage of votes that match
            }

    if not holistic_matches:
        return None

    # Select the holistic color with highest match percentage
    best_part = max(holistic_matches.items(), key=lambda x: x[1]['score'])
    part_name, match_data = best_part

    return {
        'rgb': match_data['rgb'],
        'part_name': part_name,
        'match_score': match_data['score'],
        'match_count': match_data['match_count']
    }


def voting_consensus(rgb_list, holistic_colors=None):
    """
    Find consensus RGB from a list of RGB votes using voting mechanism.

    Args:
        rgb_list: List of RGB tuples/lists (may contain None values)
        holistic_colors: Optional dict of holistic colors from Step 1 for better selection

    Returns:
        dict: {
            'rgb_final': [R, G, B],
            'consensus_score': float (0-1),
            'method': str (description of method used),
            'rgb_votes': list of valid RGB votes
        }
    """
    # Filter out None values
    valid_rgbs = [rgb for rgb in rgb_list if rgb is not None]
    
    if len(valid_rgbs) == 0:
        return {
            'rgb_final': None,
            'consensus_score': 0.0,
            'method': 'no_valid_votes',
            'rgb_votes': []
        }
    
    if len(valid_rgbs) == 1:
        return {
            'rgb_final': list(valid_rgbs[0]),
            'consensus_score': 1.0,
            'method': 'single_vote',
            'rgb_votes': [list(valid_rgbs[0])]
        }
    
    # Calculate pairwise distances
    distances = []
    for i in range(len(valid_rgbs)):
        for j in range(i + 1, len(valid_rgbs)):
            dist = calculate_rgb_distance(valid_rgbs[i], valid_rgbs[j])
            distances.append(dist)
    
    avg_distance = sum(distances) / len(distances) if distances else 0

    # If all votes are extremely similar (avg distance < 15), use average
    if avg_distance < 15:
        avg_rgb = [
            int(sum(rgb[i] for rgb in valid_rgbs) / len(valid_rgbs))
            for i in range(3)
        ]
        return {
            'rgb_final': avg_rgb,
            'consensus_score': 1.0 - (avg_distance / 100.0),  # Normalize to 0-1
            'method': 'voting_average',
            'rgb_votes': [list(rgb) for rgb in valid_rgbs]
        }

    # For moderate differences, use holistic context if available
    if holistic_colors and avg_distance >= 15 and avg_distance < 40:
        best_match = find_best_holistic_match_for_votes(valid_rgbs, holistic_colors)
        if best_match:
            return {
                'rgb_final': best_match['rgb'],
                'consensus_score': 0.8,  # High confidence with holistic guidance
                'method': 'holistic_guided_selection',
                'rgb_votes': [list(rgb) for rgb in valid_rgbs],
                'holistic_match': best_match
            }
    
    # Find RGB with minimum average distance to others (most consistent)
    min_avg_dist = float('inf')
    best_rgb_idx = 0
    
    for i in range(len(valid_rgbs)):
        avg_dist = sum(calculate_rgb_distance(valid_rgbs[i], valid_rgbs[j]) 
                      for j in range(len(valid_rgbs)) if i != j) / (len(valid_rgbs) - 1)
        if avg_dist < min_avg_dist:
            min_avg_dist = avg_dist
            best_rgb_idx = i
    
    # Check if there's a clear outlier (3 similar + 1 different)
    if len(valid_rgbs) == 4:
        best_rgb = valid_rgbs[best_rgb_idx]
        distances_to_best = [calculate_rgb_distance(best_rgb, rgb) for rgb in valid_rgbs]
        distances_to_best.sort()
        
        # If 3 are similar and 1 is outlier (distance > 50)
        if distances_to_best[2] < 40 and distances_to_best[3] > 50:
            # Use average of the 3 similar ones
            similar_rgbs = [rgb for rgb in valid_rgbs 
                           if calculate_rgb_distance(best_rgb, rgb) < 40]
            avg_rgb = [
                int(sum(rgb[i] for rgb in similar_rgbs) / len(similar_rgbs))
                for i in range(3)
            ]
            return {
                'rgb_final': avg_rgb,
                'consensus_score': 0.75,
                'method': 'voting_3_of_4',
                'rgb_votes': [list(rgb) for rgb in valid_rgbs]
            }
    
    # If all are different (avg distance > 50), flag for review but use best
    if avg_distance > 50:
        return {
            'rgb_final': list(valid_rgbs[best_rgb_idx]),
            'consensus_score': 0.5,
            'method': 'voting_best_of_diverse',
            'rgb_votes': [list(rgb) for rgb in valid_rgbs]
        }
    
    # Default: use the most consistent RGB
    return {
        'rgb_final': list(valid_rgbs[best_rgb_idx]),
        'consensus_score': 1.0 - (min_avg_dist / 100.0),
        'method': 'voting_most_consistent',
        'rgb_votes': [list(rgb) for rgb in valid_rgbs]
    }


def apply_common_sense_color(part_name, part_type, typical_colors):
    """
    Generate RGB color using common sense reasoning for small/occluded parts.
    
    Args:
        part_name: Functional name of the part (e.g., "chair_leg")
        part_type: Type/description of the part
        typical_colors: List of typical color names for this part type
    
    Returns:
        tuple: (R, G, B) as integers
    """
    # Material type inference based on part name and type
    part_lower = part_name.lower()
    type_lower = str(part_type).lower() if part_type else ""
    
    # Determine material type
    if any(word in part_lower or word in type_lower for word in ['leg', 'frame', 'support', 'base']):
        # Likely wood or metal
        if 'metal' in type_lower or 'steel' in type_lower:
            # Metal: grays, silvers
            return (120, 120, 120)  # Medium gray
        else:
            # Wood: browns, tans
            return (139, 90, 43)  # Saddle brown
    elif any(word in part_lower or word in type_lower for word in ['back', 'seat', 'cushion', 'fabric', 'upholstery']):
        # Fabric: grays, blues, browns
        if 'gray' in str(typical_colors).lower() or 'grey' in str(typical_colors).lower():
            return (128, 128, 128)  # Gray
        elif 'blue' in str(typical_colors).lower():
            return (100, 130, 180)  # Blue-gray
        else:
            return (150, 120, 100)  # Beige-brown
    elif any(word in part_lower or word in type_lower for word in ['arm', 'handle']):
        # Could be wood, metal, or fabric
        if 'wood' in type_lower:
            return (139, 90, 43)  # Brown
        elif 'metal' in type_lower:
            return (100, 100, 100)  # Gray
        else:
            return (120, 100, 80)  # Warm brown
    else:
        # Default: use typical colors if provided
        if typical_colors:
            color_str = str(typical_colors[0]).lower() if typical_colors else ""
            if 'brown' in color_str or 'wood' in color_str:
                return (139, 90, 43)  # Brown
            elif 'gray' in color_str or 'grey' in color_str:
                return (128, 128, 128)  # Gray
            elif 'black' in color_str:
                return (30, 30, 30)  # Dark gray/black
            elif 'white' in color_str:
                return (240, 240, 240)  # Light gray/white
    
    # Ultimate fallback: neutral brown-gray
    return (120, 100, 80)


# ============================================================================
# Step 1: Holistic Color Analysis
# ============================================================================

def query_holistic_colors(folder_path, captured_images_folder, api_key, type_str=None):
    """
    Step 1: Analyze 4 captured images to get holistic color information.
    
    Args:
        folder_path: Path to folder for saving results
        captured_images_folder: Path to captured_images folder containing frame images
        api_key: Not used (kept for compatibility)
        type_str: Object type (e.g., "Chair")
    
    Returns:
        dict: Holistic color analysis results
    """
    output_file = f"{folder_path}/gpt4_query/holistic_colors.json"
    os.makedirs(os.path.split(output_file)[0], exist_ok=True)
    
    # Load existing results if they exist
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            existing_data = json.load(f)
            if existing_data and 'color_details' in existing_data:
                print("Holistic colors already exist, skipping Step 1.")
                return existing_data
    
    # Find frame images (frame_*.jpg)
    frame_images = []
    if os.path.exists(captured_images_folder):
        for filename in sorted(os.listdir(captured_images_folder)):
            if filename.startswith('frame_') and filename.endswith('.jpg'):
                frame_images.append(os.path.join(captured_images_folder, filename))
    
    # Select 4 frames (prefer specific ones if available, otherwise first 4)
    preferred_frames = ['frame_12.jpg', 'frame_14.jpg', 'frame_15.jpg', 'frame_21.jpg']
    selected_frames = []
    
    for pref in preferred_frames:
        pref_path = os.path.join(captured_images_folder, pref)
        if pref_path in frame_images:
            selected_frames.append(pref_path)
    
    # If we don't have 4 preferred frames, add others
    for frame_path in frame_images:
        if frame_path not in selected_frames and len(selected_frames) < 4:
            selected_frames.append(frame_path)
    
    if len(selected_frames) < 4:
        print(f"Warning: Only found {len(selected_frames)} frame images, using available ones.")
    
    if len(selected_frames) == 0:
        print("Error: No frame images found in captured_images folder.")
        return None
    
    # Holistic analysis prompt
    holistic_prompt = f"""You are analyzing a {type_str or "furniture object"} to identify its main colors and material composition.

IMAGES PROVIDED:
You are given {len(selected_frames)} photographs of the same {type_str or "object"} captured from different angles. These images show the {type_str or "object"} in a real room environment.

TASK:
Analyze all {len(selected_frames)} images together and identify:
1. The main colors present in this {type_str or "object"}
2. Which parts/components have which colors
3. The overall color theme

INSTRUCTIONS:
1. Examine all {len(selected_frames)} images to get a comprehensive view of the {type_str or "object"}
2. Identify major visible parts/components (e.g., legs, backrest, seat, armrests, etc.)
3. For each major part, determine:
   - The dominant color
   - Color type/description (e.g., "brown_wood", "gray_fabric", "black_metal")
   - RGB estimate
   - Confidence level (high/medium/low based on visibility)
4. Focus on parts that are clearly visible across multiple images
5. Ignore background elements - focus only on the {type_str or "object"} itself

OUTPUT FORMAT (JSON):
{{
  "object_type": "{type_str or "object"}",
  "main_colors": {{
    "part_name_1": "color_type_1",
    "part_name_2": "color_type_2",
    ...
  }},
  "color_details": {{
    "part_name_1": {{
      "color_type": "color_type_1",
      "rgb_estimate": [R, G, B],
      "confidence": "high|medium|low"
    }},
    "part_name_2": {{
      "color_type": "color_type_2",
      "rgb_estimate": [R, G, B],
      "confidence": "high|medium|low"
    }},
    ...
  }},
  "overall_theme": "brief description of overall color theme"
}}

IMPORTANT:
- Use functional part names (e.g., "chair_legs", "backrest", "seat_cushion"), not technical IDs
- Provide RGB estimates for each part
- Include confidence levels based on how clearly visible each part is
- Capture all major colors present in the {type_str or "object"}
- Output ONLY valid JSON, no additional text"""
    
    # Query with multi-image
    response, success, count = None, False, 0
    while count < 3 and not success:
        try:
            response = process_image_multi_qwen(selected_frames, holistic_prompt, max_new_tokens=1000, max_image_size=600)
            response_dict = _get_response_json(response)
            
            if 'error' in response_dict.keys():
                print("Error detected, waiting...")
                time.sleep(5)
                count += 1
                continue
            
            success = True
            break
        except Exception as e:
            print(f"Error querying holistic colors: {e}")
            count += 1
            time.sleep(5)
    
    if not success:
        print("Failed to get holistic color analysis after 3 attempts.")
        return None
    
    # Parse JSON response
    response_text = response_dict["choices"][0]["message"]["content"]
    
    # Try to extract JSON from response (might be wrapped in markdown code blocks)
    json_text = response_text
    if "```json" in response_text:
        json_text = response_text.split("```json")[1].split("```")[0].strip()
    elif "```" in response_text:
        json_text = response_text.split("```")[1].split("```")[0].strip()
    
    try:
        holistic_data = json.loads(json_text)
    except json.JSONDecodeError:
        # Try to fix common JSON issues
        json_text = re.sub(r',\s*}', '}', json_text)  # Remove trailing commas
        json_text = re.sub(r',\s*]', ']', json_text)
        try:
            holistic_data = json.loads(json_text)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            print(f"Response text: {response_text[:500]}")
            return None
    
    # Validate structure
    if 'color_details' not in holistic_data:
        holistic_data['color_details'] = {}
    if 'main_colors' not in holistic_data:
        holistic_data['main_colors'] = {}
    if 'object_type' not in holistic_data:
        holistic_data['object_type'] = type_str or "object"
    if 'overall_theme' not in holistic_data:
        holistic_data['overall_theme'] = ""
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(holistic_data, f, indent=4)
    
    # Save prompt log
    prompt_log_file = f"{folder_path}/gpt4_query/color_query_prompts_log.json"
    prompt_log = {}
    if os.path.exists(prompt_log_file):
        with open(prompt_log_file) as f:
            prompt_log = json.load(f)
    
    prompt_log['step1_holistic'] = {
        'prompt': holistic_prompt,
        'input_images': selected_frames,
        'response': response_text
    }
    
    with open(prompt_log_file, 'w') as f:
        json.dump(prompt_log, f, indent=4)
    
    return holistic_data


# ============================================================================
# Step 2: Part Identification Enhanced
# ============================================================================

def query_part_identification_enhanced(folder_path, combine_folder, api_key, type_str=None):
    """
    Step 2: Identify parts using combined images from top_4_combine folder.

    Args:
        folder_path: Path to folder for saving results
        combine_folder: Path to top_4_combine folder
        api_key: Not used (kept for compatibility)
        type_str: Object type (e.g., "Chair")

    Returns:
        dict: Part identification results mapping part IDs to part names and typical colors
    """
    output_file = f"{folder_path}/gpt4_query/part_identification.json"
    os.makedirs(os.path.split(output_file)[0], exist_ok=True)

    # Load existing results if they exist
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            existing_data = json.load(f)
            if existing_data:
                print("Part identification already exists, skipping Step 2.")
                return existing_data

    part_identification = {}

    # Get all combined images
    if not os.path.exists(combine_folder):
        print(f"Error: top_4_combine folder not found: {combine_folder}")
        return {}

    # Get all .png files in the combine folder
    combine_images = [f for f in os.listdir(combine_folder)
                     if f.endswith('.png') and os.path.isfile(os.path.join(combine_folder, f))]

    if len(combine_images) == 0:
        print(f"Warning: No .png files found in {combine_folder}")
        return {}

    # Process each combined image
    for image_file in sorted(combine_images):
        part_id = image_file.split('.')[0]  # e.g., 'solid_001' from 'solid_001.png'
        image_path = os.path.join(combine_folder, image_file)
        
        # Use segmentation prompt for part identification (single combined image)
        identification_prompt = f"""You are analyzing a segmented part from a {type_str or "3D object"}.

This is an image of a rendered 3D mesh model of {type_str or "3D object"} object, with red-highlighted segmentation masks. The image shows the rendered view with the segmentation mask, with the red area marking the region of interest. Refer to the image to understand the red segment, and describe the component and its potential materials.

IMPORTANT: The RED color you see highlighting regions in these images is a SEGMENTATION MASK that shows which part we're analyzing. DO NOT include "red" as a typical material color unless the ACTUAL MATERIAL underneath the mask is red. Focus on the TRUE material color of the part being highlighted, not the red overlay.

TASK:
1. Describe what this part appears to be (what material/object it represents)
2. Identify typical colors this part would have based on its material type and appearance

OUTPUT FORMAT (JSON):
{{
  "description": "The red-highlighted region in the segmentation mask shows .......)",
  "typical_colors": ["color1", "color2", "color3"]
}}

IMPORTANT:
- Focus on material properties and visual appearance
- This is MATERIAL SEGMENTATION, not functional segmentation
- Provide typical colors that would be reasonable for the identified material type
- Output ONLY valid JSON, no additional text"""

        # Query with single image
        response, success, count = None, False, 0
        while count < 3 and not success:
            try:
                response = process_image(image_path, api_key, identification_prompt)
                response_dict = _get_response_json(response)

                if 'error' in response_dict.keys():
                    print("Error detected, waiting...")
                    time.sleep(5)
                    count += 1
                    continue

                success = True
                break
            except Exception as e:
                print(f"Error querying part identification for {part_id}: {e}")
                count += 1
                time.sleep(5)
        
        if not success:
            print(f"Failed to identify part {part_id} after 3 attempts, skipping.")
            continue
        
        # Parse JSON response
        response_text = response_dict["choices"][0]["message"]["content"]
        
        # Try to extract JSON from response
        json_text = response_text
        if "```json" in response_text:
            json_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_text = response_text.split("```")[1].split("```")[0].strip()
        
        try:
            part_info = json.loads(json_text)
        except json.JSONDecodeError:
            # Try to fix common JSON issues
            json_text = re.sub(r',\s*}', '}', json_text)
            json_text = re.sub(r',\s*]', ']', json_text)
            try:
                part_info = json.loads(json_text)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON response for {part_id}: {e}")
                print(f"Response text: {response_text[:500]}")
                continue
        
        # Validate structure
        if 'part_name' not in part_info:
            part_info['part_name'] = part_id
        if 'description' not in part_info:
            part_info['description'] = ""
        if 'typical_colors' not in part_info:
            part_info['typical_colors'] = []
        
        part_identification[part_id] = part_info
        
        # Save progress incrementally
        with open(output_file, 'w') as f:
            json.dump(part_identification, f, indent=4)
    
    # Save prompt log
    prompt_log_file = f"{folder_path}/gpt4_query/color_query_prompts_log.json"
    prompt_log = {}
    if os.path.exists(prompt_log_file):
        with open(prompt_log_file) as f:
            prompt_log = json.load(f)
    
    prompt_log['step2_identification'] = {
        'prompt_template': identification_prompt,
        'parts_processed': list(part_identification.keys())
    }
    
    with open(prompt_log_file, 'w') as f:
        json.dump(prompt_log, f, indent=4)
    
    return part_identification


# ============================================================================
# Step 3: Per-Part Color Query with Voting
# ============================================================================

def query_part_color_with_voting(folder_path, stitched_image_path, top_9_folder, part_info, holistic_data, api_key, type_str=None):
    """
    Step 3: Query color 4 times with different images, then vote for consensus.
    Uses holistic analysis from Step 1 for better color consistency.

    Args:
        folder_path: Path to folder for saving results
        stitched_image_path: Path to stitched_image.jpg (reference image)
        top_9_folder: Path to top_9_labeled_images folder
        part_info: Dictionary from Step 2 with part identification info
        holistic_data: Dictionary from Step 1 with holistic color analysis
        api_key: Not used (kept for compatibility)
        type_str: Object type (e.g., "Chair")

    Returns:
        dict: RGB voting results for each part
    """
    output_file = f"{folder_path}/gpt4_query/rgb_voting_results.json"
    os.makedirs(os.path.split(output_file)[0], exist_ok=True)
    
    # Load existing results if they exist
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            existing_data = json.load(f)
            if existing_data:
                print("RGB voting results already exist, skipping Step 3.")
                return existing_data
    
    if not os.path.exists(stitched_image_path):
        print(f"Error: stitched_image.jpg not found at {stitched_image_path}")
        return {}
    
    rgb_voting_results = {}
    
    # Process each part
    for part_id, part_data in part_info.items():
        part_name = part_data.get('part_name', part_id)
        part_description = part_data.get('description', '')
        typical_colors = part_data.get('typical_colors', [])
        
        part_folder = os.path.join(top_9_folder, part_id)
        if not os.path.exists(part_folder):
            print(f"Warning: Part folder not found: {part_folder}, skipping.")
            continue
        
        # Get labeled images for this part
        labeled_images = []
        for filename in sorted(os.listdir(part_folder)):
            if filename.endswith('_labeled.png'):
                labeled_images.append(os.path.join(part_folder, filename))
        
        if len(labeled_images) < 4:
            print(f"Warning: Part {part_id} has only {len(labeled_images)} labeled images.")
        
        # Select 4 different images for voting (use first 4 or all available)
        selected_images = labeled_images[:4] if len(labeled_images) >= 4 else labeled_images
        
        if len(selected_images) == 0:
            print(f"Warning: No labeled images found for part {part_id}, using common sense.")
            # Use common sense fallback
            common_sense_rgb = apply_common_sense_color(part_name, part_description, typical_colors)
            rgb_voting_results[part_id] = {
                'part_name': part_name,
                'rgb_final': list(common_sense_rgb),
                'rgb_votes': [list(common_sense_rgb)],
                'consensus_score': 1.0,
                'confidence': 'low',
                'method': 'common_sense_fallback',
                'estimation_method': 'common_sense_reasoning',
                'visibility': 'no_images',
                'reasoning': 'No labeled images available, used common sense reasoning'
            }
            continue
        
        # Helper function to format holistic context
        def format_holistic_context():
            if not holistic_data:
                return ""
            context_parts = []
            if holistic_data.get('object_type'):
                context_parts.append(f"Object Type: {holistic_data['object_type']}")
            if holistic_data.get('overall_theme'):
                context_parts.append(f"Overall Theme: {holistic_data['overall_theme']}")
            if holistic_data.get('main_colors'):
                main_colors_str = ", ".join([f"{k}: {v}" for k, v in holistic_data['main_colors'].items()])
                if main_colors_str:
                    context_parts.append(f"Main Colors: {main_colors_str}")
            return "\n".join(f"- {part}" for part in context_parts) if context_parts else ""

        holistic_context = format_holistic_context()

        # 4 different prompts for voting (using the prompts from the plan)
        prompts = [
            f"""Color estimation for "{part_description}" segment of a {type_str or "furniture object"}.

HOLISTIC OBJECT CONTEXT (from Step 1):
{holistic_context}

INPUT IMAGES:
- Image 1: Real photographs of the {type_str or "object"} captured from multiple angles (stitched_image.jpg)
- Image 2: 3D renderings with this part highlighted in red. The red area shows the segmentation mask for this specific part.

PART INFORMATION:
- Description: {part_description}

TASK:
Extract the RGB color for the part highlighted in Image 2 (segmentation mask) by analyzing Image 1 (real photographs). The goal is to provide a faithful color description of this material segment based on how it appears in the actual photographs.

INSTRUCTIONS:
- PRIORITY: Reference the HOLISTIC OBJECT CONTEXT from Step 1 as your primary guide for color selection
- If the holistic analysis describes specific colors for this part type (e.g., "backrest: blue_fabric"), match those colors as closely as possible
- Look at Image 1 (photographs) to locate the corresponding part that matches the red-highlighted region in Image 2
- IGNORE the red segmentation mask overlay in Image 2 - this is just to show you which part to analyze
- Focus on finding the corresponding region in Image 1 (the real photograph) and extract the TRUE color of the material from there
- If the part is clearly visible in Image 1, extract the dominant color directly from the visible region in the photographs
- Ensure the extracted color aligns with the holistic object theme and main colors from Step 1
- If the part is not clearly visible or very small in Image 1, use the holistic context as your main reference, supplemented by the part description
- The holistic analysis represents the true appearance - prioritize consistency with the main colors description over generic assumptions
- Choose a color that fits the object's overall style from Step 1 and this part's material type

OUTPUT: [R, G, B]""",
            
            f"""Color estimation for "{part_description}" segment of a {type_str or "furniture object"}.

HOLISTIC OBJECT CONTEXT (from Step 1):
{holistic_context}

INPUT IMAGES:
- Image 1: Real photographs of the {type_str or "object"} captured from multiple angles (stitched_image.jpg)
- Image 2: 3D renderings with this part highlighted in red from a different angle

PART INFORMATION:
- Description: {part_description}

TASK:
Extract the RGB color for the part highlighted in Image 2 (segmentation mask) by analyzing Image 1 (real photographs). The goal is to provide a faithful color description of this material segment based on how it appears in the actual photographs.

INSTRUCTIONS:
- PRIORITY: Reference the HOLISTIC OBJECT CONTEXT from Step 1 as your primary guide for color selection
- If the holistic analysis describes specific colors for this part type (e.g., "backrest: blue_fabric"), match those colors as closely as possible
- Look at Image 1 (photographs) to locate the corresponding part that matches the red-highlighted region in Image 2
- IGNORE the red segmentation mask overlay in Image 2 - this is just to show you which part to analyze
- Focus on finding the corresponding region in Image 1 (the real photograph) and extract the TRUE color of the material from there
- If the part is clearly visible in Image 1, extract the dominant color directly from the visible region in the photographs
- Ensure the extracted color aligns with the holistic object theme and main colors from Step 1
- If the part is not clearly visible or very small in Image 1, use the holistic context as your main reference, supplemented by the part description
- The holistic analysis represents the true appearance - prioritize consistency with the main colors description over generic assumptions
- Choose a color that fits the object's overall style from Step 1 and this part's material type

OUTPUT: [R, G, B]""",
            
            f"""Color estimation for "{part_description}" segment of a {type_str or "furniture object"}.

HOLISTIC OBJECT CONTEXT (from Step 1):
{holistic_context}

INPUT IMAGES:
- Image 1: Real-world photographs (stitched_image.jpg)
- Image 2: 3D mesh rendering with this part highlighted in red

PART INFORMATION:
- Description: {part_description}

TASK:
Extract the RGB color for the part highlighted in Image 2 (segmentation mask) by analyzing Image 1 (real photographs). The goal is to provide a faithful color description of this material segment based on how it appears in the actual photographs.

INSTRUCTIONS:
- PRIORITY: Reference the HOLISTIC OBJECT CONTEXT from Step 1 as your primary guide for color selection
- If the holistic analysis describes specific colors for this part type (e.g., "backrest: blue_fabric"), match those colors as closely as possible
- Look at Image 1 (photographs) to locate the corresponding part that matches the red-highlighted region in Image 2
- IGNORE the red segmentation mask overlay in Image 2 - this is just to show you which part to analyze
- Focus on finding the corresponding region in Image 1 (the real photograph) and extract the TRUE color of the material from there
- If the part is clearly visible in Image 1, extract the dominant color directly from the visible region in the photographs
- Ensure the extracted color aligns with the holistic object theme and main colors from Step 1
- If the part is not clearly visible or very small in Image 1, use the holistic context as your main reference, supplemented by the part description
- The holistic analysis represents the true appearance - prioritize consistency with the main colors description over generic assumptions
- Choose a color that fits the object's overall style from Step 1 and this part's material type

OUTPUT: [R, G, B]""",
            
            f"""Final color estimation for "{part_description}" segment of a {type_str or "furniture object"}.

HOLISTIC OBJECT CONTEXT (from Step 1):
{holistic_context}

INPUT IMAGES:
- Image 1: Real-world photographs (stitched_image.jpg) showing multiple views of the real {type_str or "object"}
- Image 2: 3D rendering with this part highlighted in red

PART INFORMATION:
- Description: {part_description}

TASK:
Extract the RGB color for the part highlighted in Image 2 (segmentation mask) by analyzing Image 1 (real photographs). The goal is to provide a faithful color description of this material segment based on how it appears in the actual photographs.

INSTRUCTIONS:
- PRIORITY: Reference the HOLISTIC OBJECT CONTEXT from Step 1 as your primary guide for color selection
- If the holistic analysis describes specific colors for this part type (e.g., "backrest: blue_fabric"), match those colors as closely as possible
- Look at Image 1 (photographs) to locate the corresponding part that matches the red-highlighted region in Image 2
- IGNORE the red segmentation mask overlay in Image 2 - this is just to show you which part to analyze
- Focus on finding the corresponding region in Image 1 (the real photograph) and extract the TRUE color of the material from there
- If the part is clearly visible in Image 1, extract the dominant color directly from the visible region in the photographs
- Ensure the extracted color aligns with the holistic object theme and main colors from Step 1
- If the part is not clearly visible or very small in Image 1, use the holistic context as your main reference, supplemented by the part description
- The holistic analysis represents the true appearance - prioritize consistency with the main colors description over generic assumptions
- Choose a color that fits the object's overall style from Step 1 and this part's material type

OUTPUT: [R, G, B]"""
        ]
        
        # Query 4 times with different images
        rgb_votes = []
        for query_idx, (image_path, prompt) in enumerate(zip(selected_images[:4], prompts[:4])):
            # Combine stitched image + part image for query
            image_paths = [stitched_image_path, image_path]
            
            response, success, count = None, False, 0
            while count < 3 and not success:
                try:
                    response = process_image_multi_qwen(image_paths, prompt, max_new_tokens=300, max_image_size=600)
                    response_dict = _get_response_json(response)
                    
                    if 'error' in response_dict.keys():
                        print("Error detected, waiting...")
                        time.sleep(5)
                        count += 1
                        continue
                    
                    success = True
                    break
                except Exception as e:
                    print(f"Error querying color for {part_id} (query {query_idx+1}): {e}")
                    count += 1
                    time.sleep(5)
            
            if not success:
                print(f"Failed to get color for {part_id} query {query_idx+1}, skipping this vote.")
                continue
            
            # Parse RGB from response
            response_text = response_dict["choices"][0]["message"]["content"]
            rgb = parse_rgb_from_response(response_text)
            
            if rgb is None:
                print(f"Warning: Could not parse RGB from response for {part_id} query {query_idx+1}, using common sense.")
                rgb = apply_common_sense_color(part_name, part_description, typical_colors)
            
            rgb_votes.append(rgb)
        
        # If we have no valid votes, use common sense
        if len(rgb_votes) == 0:
            common_sense_rgb = apply_common_sense_color(part_name, part_description, typical_colors)
            rgb_votes = [common_sense_rgb]
        
        # Apply voting consensus with holistic guidance
        consensus_result = voting_consensus(rgb_votes, holistic_data)
        
        # Determine confidence and visibility
        if consensus_result['consensus_score'] >= 0.9:
            confidence = 'high'
        elif consensus_result['consensus_score'] >= 0.7:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        # Check if common sense was likely used (all votes very similar to common sense)
        common_sense_rgb = apply_common_sense_color(part_name, part_description, typical_colors)
        avg_dist_to_common_sense = sum(calculate_rgb_distance(rgb, common_sense_rgb) for rgb in rgb_votes) / len(rgb_votes)
        
        if avg_dist_to_common_sense < 20 and consensus_result['consensus_score'] < 0.8:
            estimation_method = 'common_sense_reasoning'
            visibility = 'likely_small_or_occluded'
        else:
            estimation_method = 'visual_extraction'
            visibility = 'clear' if consensus_result['consensus_score'] >= 0.8 else 'partially_visible'
        
        rgb_voting_results[part_id] = {
            'part_name': part_name,
            'rgb_final': consensus_result['rgb_final'],
            'rgb_votes': consensus_result['rgb_votes'],
            'consensus_score': consensus_result['consensus_score'],
            'confidence': confidence,
            'method': consensus_result['method'],
            'estimation_method': estimation_method,
            'visibility': visibility
        }
        
        # Save progress incrementally
        with open(output_file, 'w') as f:
            json.dump(rgb_voting_results, f, indent=4)
    
    # Save prompt log
    prompt_log_file = f"{folder_path}/gpt4_query/color_query_prompts_log.json"
    prompt_log = {}
    if os.path.exists(prompt_log_file):
        with open(prompt_log_file) as f:
            prompt_log = json.load(f)
    
    prompt_log['step3_voting'] = {
        'prompt_templates': prompts[:1],  # Save first prompt as template
        'parts_processed': list(rgb_voting_results.keys())
    }
    
    with open(prompt_log_file, 'w') as f:
        json.dump(prompt_log, f, indent=4)
    
    return rgb_voting_results


# ============================================================================
# Step 4: Validation & Iterative Refinement
# ============================================================================

def validate_and_refine_colors(folder_path, holistic_colors_path, voting_results_path, stitched_image_path, top_9_folder, part_info, api_key, type_str=None, max_iterations=2):
    """
    Step 4: HOLISTIC-FIRST Color Validation & Refinement

    PHASE 1: Align part colors with holistic colors - prefer holistic when inconsistent
    PHASE 2: Ensure all holistic colors are represented in final part colors

    Args:
        folder_path: Path to folder for saving results
        holistic_colors_path: Path to holistic_colors.json from Step 1
        voting_results_path: Path to rgb_voting_results.json from Step 3
        stitched_image_path: Path to stitched_image.jpg (for verification)
        top_9_folder: Path to top_9_labeled_images folder (for verification)
        part_info: Dictionary from Step 2 with part identification info
        api_key: Not used (kept for compatibility)
        type_str: Object type (e.g., "Chair")
        max_iterations: Maximum number of refinement iterations (default: 2)

    Returns:
        dict: Final RGB colors and validation report
    """
    final_output_file = f"{folder_path}/gpt4_query/final_rgb_colors.json"
    validation_report_file = f"{folder_path}/gpt4_query/validation_report.json"
    os.makedirs(os.path.split(final_output_file)[0], exist_ok=True)

    # Load Step 1 and Step 3 results
    try:
        with open(holistic_colors_path, 'r') as f:
            holistic_data = json.load(f)
    except FileNotFoundError:
        print(f"Warning: holistic_colors.json not found at {holistic_colors_path}")
        holistic_data = {}

    try:
        with open(voting_results_path, 'r') as f:
            voting_results = json.load(f)
    except FileNotFoundError:
        print(f"Error: rgb_voting_results.json not found at {voting_results_path}")
        return {}

    # Extract holistic color information
    step1_color_details = holistic_data.get('color_details', {})
    step1_main_colors = holistic_data.get('main_colors', {})
    overall_theme = holistic_data.get('overall_theme', '')

    # Get all unique colors from holistic analysis
    holistic_rgb_colors = set()
    for part_name, color_data in step1_color_details.items():
        rgb = color_data.get('rgb_estimate')
        if rgb and isinstance(rgb, list) and len(rgb) == 3:
            holistic_rgb_colors.add(tuple(rgb))

    print(f"📊 Holistic analysis found {len(holistic_rgb_colors)} unique colors")
    print(f"🎨 Overall theme: {overall_theme}")

    # Create mapping: part_name -> part_id
    part_name_to_id = {}
    for part_id, part_data in part_info.items():
        part_name = part_data.get('part_name', '').lower()
        part_name_to_id[part_name] = part_id

    # Also try partial matches (e.g., "chair_leg" matches "chair_legs")
    def find_matching_part_id(step1_part_name):
        step1_lower = step1_part_name.lower()
        # Exact match
        if step1_lower in part_name_to_id:
            return part_name_to_id[step1_lower]
        # Partial match
        for part_name, part_id in part_name_to_id.items():
            if step1_lower in part_name or part_name in step1_lower:
                return part_id
        return None

    # ===== HELPER FUNCTIONS =====

    def find_best_holistic_color_match(per_part_rgb, holistic_details, part_description=None, api_key=None, object_type=None, folder_path=None):
        """Find the best matching holistic color for a per-part color using LLM semantic mapping."""
        if not holistic_details or not part_description:
            # Fallback to color-only matching
            return find_best_holistic_color_match_color_only(per_part_rgb, holistic_details)

        # Use LLM to determine semantic mapping
        if api_key:
            try:
                return find_best_holistic_color_match_with_llm(per_part_rgb, holistic_details, part_description, api_key, object_type)
            except Exception as e:
                print(f"Warning: LLM semantic mapping failed: {e}, falling back to color-only matching")

        # Fallback to color-only matching
        return find_best_holistic_color_match_color_only(per_part_rgb, holistic_details)


    def find_best_holistic_color_match_color_only(per_part_rgb, holistic_details):
        """Fallback: Find best holistic color match using only RGB distance."""
        best_match = None
        min_distance = float('inf')

        for part_name, color_data in holistic_details.items():
            holistic_rgb = color_data.get('rgb_estimate')
            if holistic_rgb and isinstance(holistic_rgb, list) and len(holistic_rgb) == 3:
                distance = calculate_rgb_distance(holistic_rgb, per_part_rgb)
                if distance < min_distance:
                    min_distance = distance
                    best_match = {
                        'rgb': holistic_rgb,
                        'part_name': part_name,
                        'distance': distance,
                        'color_type': color_data.get('color_type', '')
                    }

        return best_match


    def find_best_holistic_color_match_with_llm(per_part_rgb, holistic_details, part_description, api_key, object_type, folder_path):
        """Use Qwen LLM to semantically map part description to holistic parts."""

        # Build holistic parts description
        holistic_parts_text = []
        for part_name, color_data in holistic_details.items():
            rgb = color_data.get('rgb_estimate', [])
            color_type = color_data.get('color_type', 'unknown')
            holistic_parts_text.append(f"- {part_name}: {color_type} color RGB{rgb}")

        holistic_parts_str = "\n".join(holistic_parts_text)

        prompt = f"""You are mapping a segmented part from a {object_type} to the holistic parts identified in Step 1.

SEGMENTED PART DESCRIPTION:
{part_description}

HOLISTIC PARTS FROM STEP 1:
{holistic_parts_str}

TASK:
Determine which holistic part this segmented part most likely corresponds to. Consider:
1. Functional similarity (what is this part used for?)
2. Physical location (where is this part located on the object?)
3. Material description (what material is this part made of?)
4. Semantic meaning (what does this part represent?)

OUTPUT FORMAT (JSON):
{{
  "matched_holistic_part": "exact_part_name_from_list",
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation of why this mapping makes sense"
}}

IMPORTANT:
- Choose ONLY from the holistic parts listed above
- If no good match exists, choose the closest functional equivalent
- Output ONLY valid JSON, no additional text"""

        try:
            # Use Qwen for semantic mapping - use a dummy image since Qwen requires images
            # We'll use the stitched image as a reference
            dummy_image_path = f"{folder_path}/../captured_images/stitched_image.jpg"

            if os.path.exists(dummy_image_path):
                from .qwen_query_materials import process_image_qwen

                response = process_image_qwen(dummy_image_path, prompt, max_new_tokens=300, max_image_size=600)
                response_dict = _get_response_json(response)

                if 'choices' in response_dict and len(response_dict['choices']) > 0:
                    result_text = response_dict['choices'][0]['message']['content'].strip()

                    # Parse JSON response
                    import json
                    result = json.loads(result_text)
                else:
                    raise Exception("Invalid Qwen response format")
            else:
                raise Exception("No reference image available for semantic mapping")

        except Exception as e:
            print(f"Qwen semantic mapping failed: {e}")
            # Don't raise - fall back to color matching

            matched_part = result.get('matched_holistic_part')
            confidence = result.get('confidence', 0.5)

            # Find the matched part's data
            if matched_part and matched_part in holistic_details:
                color_data = holistic_details[matched_part]
                holistic_rgb = color_data.get('rgb_estimate')

                return {
                    'rgb': holistic_rgb,
                    'part_name': matched_part,
                    'distance': calculate_rgb_distance(holistic_rgb, per_part_rgb) if holistic_rgb else float('inf'),
                    'confidence': confidence,
                    'reasoning': result.get('reasoning', ''),
                    'method': 'llm_semantic_mapping'
                }

        except Exception as e:
            print(f"Qwen semantic mapping failed: {e}")

        # Fallback to color-only matching
        return find_best_holistic_color_match_color_only(per_part_rgb, holistic_details)

    def verify_color_with_images(holistic_rgb, per_part_rgb, part_description, holistic_part,
                                stitched_image_path, part_id, top_9_folder, object_type, api_key):
        """Use VLM to verify which color to prefer when holistic and per-part disagree."""

        # Get part image for verification
        part_image_path = None
        if os.path.exists(top_9_folder):
            part_folder = os.path.join(top_9_folder, part_id)
            if os.path.exists(part_folder):
                for filename in sorted(os.listdir(part_folder)):
                    if filename.endswith('_labeled.png'):
                        part_image_path = os.path.join(part_folder, filename)
                        break

        if not part_image_path and not os.path.exists(stitched_image_path):
            return {'prefer_holistic': True, 'reasoning': 'No images available for verification'}

        verification_prompt = f"""You are comparing two color estimates for a {object_type} part.

HOLISTIC ANALYSIS (Step 1 - Object Level):
- Estimated color: RGB{holistic_rgb}
- From holistic part: "{holistic_part}"
- Overall object theme: "{overall_theme}"

PER-PART ANALYSIS (Step 3 - Detail Level):
- Estimated color: RGB{per_part_rgb}
- Part description: "{part_description}"

IMAGES PROVIDED:
- Image 1: Full object view (stitched image)
- Image 2: Specific part segmentation

TASK:
Compare the two color estimates and determine which one better represents the actual appearance of this part in the real object. Consider:

1. VISUAL CONSISTENCY: Which color better matches what you see in the images?
2. OBJECT COHERENCE: Which choice maintains better color harmony with the overall object?
3. PART FUNCTION: Does the part's purpose suggest it should match holistic or be distinct?

OUTPUT FORMAT (JSON):
{{
  "preferred_color": "holistic" or "per_part",
  "reasoning": "Brief explanation of your choice",
  "confidence": 0.0-1.0
}}

Be honest about which color actually appears in the images, even if it contradicts the holistic analysis."""

        try:
            images_to_use = []
            if os.path.exists(stitched_image_path):
                images_to_use.append(stitched_image_path)
            if part_image_path:
                images_to_use.append(part_image_path)

            if images_to_use:
                response = process_image_multi_qwen(images_to_use, verification_prompt, max_new_tokens=300, max_image_size=600)
                response_dict = _get_response_json(response)

                if 'choices' in response_dict and len(response_dict['choices']) > 0:
                    content = response_dict['choices'][0]['message']['content']

                    # Try to parse JSON
                    try:
                        result = json.loads(content)
                        prefer_holistic = result.get('preferred_color') == 'holistic'
                        return {
                            'prefer_holistic': prefer_holistic,
                            'reasoning': result.get('reasoning', 'VLM verification'),
                            'confidence': result.get('confidence', 0.5)
                        }
                    except json.JSONDecodeError:
                        # Fallback: check if response mentions preferring holistic
                        content_lower = content.lower()
                        if 'holistic' in content_lower and 'prefer' in content_lower:
                            return {'prefer_holistic': True, 'reasoning': 'VLM indicated preference for holistic'}
                        else:
                            return {'prefer_holistic': False, 'reasoning': 'VLM indicated preference for per-part'}

        except Exception as e:
            print(f"Warning: Color verification failed: {e}")

        # Default fallback: prefer holistic for consistency
        return {'prefer_holistic': True, 'reasoning': 'Verification failed, defaulting to holistic for consistency'}

    def find_best_part_for_missing_color(missing_color, source_holistic_part, current_final_colors, part_info, holistic_details):
        """Find the best part to assign a missing holistic color to."""

        best_assignment = None
        best_score = -1

        for part_id, current_color in current_final_colors.items():
            if current_color is None:
                # Part with no color assigned - perfect candidate
                part_data = part_info.get(part_id, {})
                part_description = part_data.get('description', '').lower()

                # Check if part description suggests it could have this color
                holistic_color_data = None
                for h_part, h_data in holistic_details.items():
                    if tuple(h_data.get('rgb_estimate', [])) == tuple(missing_color):
                        holistic_color_data = h_data
                        break

                if holistic_color_data:
                    color_type = holistic_color_data.get('color_type', '').lower()

                    # Score based on semantic match
                    score = 0
                    if color_type in part_description:
                        score += 3  # Strong semantic match
                    if any(word in part_description for word in color_type.split()):
                        score += 2  # Partial semantic match
                    if source_holistic_part and source_holistic_part.lower() in part_id.lower():
                        score += 1  # Part name similarity

                    if score > best_score:
                        best_score = score
                        best_assignment = {
                            'part_id': part_id,
                            'reasoning': f'Semantic match (score: {score}) - {color_type} likely belongs on {part_description[:50]}...'
                        }

        return best_assignment

    # ===== PHASE 1: HOLISTIC ALIGNMENT =====
    # For each part, align with holistic colors - prefer holistic when inconsistent
    print(f"\n🎯 PHASE 1: Holistic Alignment")
    validation_results = {}
    final_colors = {}

    for part_id, voting_data in voting_results.items():
        part_name = voting_data.get('part_name', part_id)
        step3_rgb = voting_data.get('rgb_final')
        part_description = part_info.get(part_id, {}).get('description', '')

        if step3_rgb is None:
            validation_results[part_id] = {
                'status': 'no_color',
                'flag_level': 3,
                'action': 'use_holistic_fallback',
                'final_color': None  # Will be assigned in Phase 2
            }
            continue

        # Find best matching holistic color using LLM semantic mapping
        best_holistic_match = find_best_holistic_color_match(step3_rgb, step1_color_details, part_description, api_key, type_str, folder_path)
        holistic_rgb = best_holistic_match['rgb'] if best_holistic_match else None
        holistic_part = best_holistic_match['part_name'] if best_holistic_match else None

        if holistic_rgb is None:
            # No holistic color found - use per-part result
            validation_results[part_id] = {
                'step1_color': None,
                'step3_color': step3_rgb,
                'rgb_distance': None,
                'status': 'no_holistic_match',
                'flag_level': 1,
                'action': 'use_per_part',
                'reasoning': 'No matching holistic color found'
            }
            final_colors[part_id] = step3_rgb
        else:
            # Compare distances and decide
            rgb_distance = calculate_rgb_distance(holistic_rgb, step3_rgb)

            # HOLISTIC-FIRST DECISION LOGIC
            if rgb_distance < 25:
                # Very close match - use per-part for precision
                decision = 'use_per_part'
                final_rgb = step3_rgb
                reasoning = f'Per-part color very close to holistic (distance: {rgb_distance})'
            elif rgb_distance < 45:
                # Moderate difference - verify with image context
                verification_result = verify_color_with_images(
                    holistic_rgb, step3_rgb, part_description, holistic_part,
                    stitched_image_path, part_id, top_9_folder, type_str, api_key
                )
                if verification_result.get('prefer_holistic', False):
                    decision = 'use_holistic'
                    final_rgb = holistic_rgb
                    reasoning = f'Image verification favors holistic: {verification_result.get("reasoning", "")}'
                else:
                    decision = 'use_per_part'
                    final_rgb = step3_rgb
                    reasoning = f'Image verification favors per-part: {verification_result.get("reasoning", "")}'
            else:
                # Large difference - prefer holistic
                decision = 'use_holistic'
                final_rgb = holistic_rgb
                reasoning = f'Large color difference ({rgb_distance}), preferring holistic consistency'

            validation_results[part_id] = {
                'holistic_color': holistic_rgb,
                'per_part_color': step3_rgb,
                'final_color': final_rgb,
                'rgb_distance': rgb_distance,
                'decision': decision,
                'reasoning': reasoning,
                'holistic_part': holistic_part,
                'flag_level': 2 if decision == 'use_holistic' else 0
            }

            final_colors[part_id] = final_rgb

    # ===== PHASE 2: COVERAGE VALIDATION =====
    # Ensure all holistic colors are represented in final part colors
    print(f"\n🔍 PHASE 2: Coverage Validation")

    final_rgb_colors = set(tuple(color) for color in final_colors.values() if color is not None)
    missing_holistic_colors = holistic_rgb_colors - final_rgb_colors

    if missing_holistic_colors:
        print(f"⚠️  Found {len(missing_holistic_colors)} holistic colors not represented in parts")

        for missing_color in missing_holistic_colors:
            # Find which holistic part this color belonged to
            source_part = None
            for part_name, color_data in step1_color_details.items():
                rgb = color_data.get('rgb_estimate')
                if rgb and tuple(rgb) == missing_color:
                    source_part = part_name
                    break

            # Find best part to assign this color to
            best_part_assignment = find_best_part_for_missing_color(
                missing_color, source_part, final_colors, part_info, step1_color_details
            )

            if best_part_assignment:
                part_id = best_part_assignment['part_id']
                print(f"  🔄 Assigning missing holistic color {missing_color} to part {part_id}")
                print(f"      Reason: {best_part_assignment['reasoning']}")

                # Update the part's color
                final_colors[part_id] = list(missing_color)

                # Update validation record
                if part_id in validation_results:
                    validation_results[part_id]['final_color'] = list(missing_color)
                    validation_results[part_id]['decision'] = 'assigned_missing_holistic'
                    validation_results[part_id]['reasoning'] += f" | Assigned missing holistic color from {source_part}"
                else:
                    validation_results[part_id] = {
                        'holistic_color': list(missing_color),
                        'per_part_color': None,
                        'final_color': list(missing_color),
                        'decision': 'assigned_missing_holistic',
                        'reasoning': f'Assigned missing holistic color from {source_part}'
                    }
    else:
        print(f"✅ All holistic colors are represented in final part colors")

    # ===== FINAL OUTPUT =====
    # Create validation report with holistic-first results
    validation_report = {
        'phase1_holistic_alignment': {
            'holistic_colors_analyzed': len(holistic_rgb_colors),
            'parts_processed': len(validation_results),
            'holistic_preferred': sum(1 for v in validation_results.values() if v.get('decision') == 'use_holistic'),
            'per_part_kept': sum(1 for v in validation_results.values() if v.get('decision') == 'use_per_part'),
            'verifications_performed': sum(1 for v in validation_results.values() if 'reasoning' in v and 'Image verification' in v.get('reasoning', ''))
        },
        'phase2_coverage_validation': {
            'missing_holistic_colors': len(missing_holistic_colors),
            'colors_assigned': len(missing_holistic_colors) - (len(missing_holistic_colors) - sum(1 for v in validation_results.values() if v.get('decision') == 'assigned_missing_holistic')),
            'final_coverage_complete': len(missing_holistic_colors) == 0
        },
        'validation_results': validation_results,
        'holistic_colors': list(holistic_rgb_colors),
        'final_colors': final_colors,
        'overall_confidence': 'high' if all(v.get('flag_level', 0) <= 1 for v in validation_results.values() if v.get('flag_level') is not None) else 'medium'
    }
    
    # Save results
    with open(final_output_file, 'w') as f:
        json.dump(final_colors, f, indent=4)
    
    with open(validation_report_file, 'w') as f:
        json.dump(validation_report, f, indent=4)
    
    # Update voting results file with refined colors
    with open(voting_results_path, 'w') as f:
        json.dump(voting_results, f, indent=4)
    
    return {
        'final_colors': final_colors,
        'validation_report': validation_report
    }


# ============================================================================
# Backward Compatibility: Convert final_rgb_colors.json to rgb_only.json format
# ============================================================================

def create_rgb_only_compatibility(folder_path):
    """
    Create rgb_only.json from final_rgb_colors.json for backward compatibility.
    This allows Material_refinements.py to work with the new format.
    
    Args:
        folder_path: Path to folder containing gpt4_query subfolder
    
    Returns:
        bool: True if conversion was successful, False otherwise
    """
    final_rgb_file = f"{folder_path}/gpt4_query/final_rgb_colors.json"
    rgb_only_file = f"{folder_path}/gpt4_query/rgb_only.json"
    
    # Check if final_rgb_colors.json exists
    if not os.path.exists(final_rgb_file):
        return False
    
    try:
        # Load final_rgb_colors.json
        with open(final_rgb_file, 'r') as f:
            final_colors = json.load(f)
        
        # Convert format: part_id -> RGB list -> RGB string "[R, G, B]"
        rgb_only_dict = {}
        for part_id, rgb_list in final_colors.items():
            if isinstance(rgb_list, list) and len(rgb_list) == 3:
                rgb_only_dict[part_id] = f"[{rgb_list[0]}, {rgb_list[1]}, {rgb_list[2]}]"
            else:
                # Handle case where RGB might already be a string
                rgb_only_dict[part_id] = str(rgb_list)
        
        # Save as rgb_only.json
        with open(rgb_only_file, 'w') as f:
            json.dump(rgb_only_dict, f, indent=4)
        
        return True
    except Exception as e:
        print(f"Error creating rgb_only.json compatibility file: {e}")
        return False


def query_part_color(folder_path, stitch_image_folder, api_key, type_str=None):

    color_to_save = f"{folder_path}/gpt4_query/rgb_only.json"
    os.makedirs(os.path.split(color_to_save)[0], exist_ok=True)

    if not os.path.exists(color_to_save):
        with open(color_to_save, 'w') as f:
            json.dump({}, f, indent=4)
    with open(color_to_save) as f:
        color_to_save_dict = json.load(f)

    previous_query = folder_path +  "/gpt4_query"
    segmentation_infor = os.path.join(previous_query, "segmentation.json")
    with open(segmentation_infor, 'r') as file:
        segmentation_info = json.load(file)

    # Extract descriptions
    descriptions = {
        key: value["choices"][0]["message"]["content"] for key, value in segmentation_info.items()
    }

    # Create a log file for prompts
    prompt_log_file = f"{folder_path}/gpt4_query/color_extraction_prompts.json"
    prompt_log = {}
    if os.path.exists(prompt_log_file):
        with open(prompt_log_file) as f:
            prompt_log = json.load(f)

    for name_of_part, description in descriptions.items():
        if name_of_part in color_to_save_dict:
            print(f"Skipping {name_of_part}, data already exists.")
            continue
        # Use part mask directly
        part_image = os.path.join(stitch_image_folder, f"{name_of_part}.png")


        prompt = color_information_extraction.format(type_str, description)

        # Save prompt to log
        prompt_log[name_of_part] = {
            "input_image": part_image,
            "prompt": prompt,
            "part_description": description
        }

        # Initialize variables for retrying
        response, success, count = '', False, 0
        part_start_time = time.time()
        while count < 3 and not success:
            try:
                # Query the API
                response = process_image(part_image, api_key, prompt)
                response_dict = _get_response_json(response)
                # Check if there's an error in the response
                if 'error' in response_dict.keys():
                    print("Error detected, waiting for rate limit...")
                    time.sleep(5)
                    count += 1
                    continue
                # Success
                success = True
                # Suppress verbose output
                break
            except Exception as e:
                # Handle any unexpected exceptions and retry
                print(f"Error querying {name_of_part}: {e}")
                count += 1
                time.sleep(5)

        # Store the response if successful
        if success:
            response_dict = _get_response_json(response)
            color_to_save_dict[name_of_part] = response_dict["choices"][0]["message"]["content"]
            
            # Save response to prompt log
            prompt_log[name_of_part]["response"] = response_dict["choices"][0]["message"]["content"]
            prompt_log[name_of_part]["response_time_seconds"] = time.time() - part_start_time
            
            # Suppress verbose output
            with open(color_to_save, 'w') as f:
                json.dump(color_to_save_dict, f, indent=4)
    
    # Save prompt log
    with open(prompt_log_file, 'w') as f:
        json.dump(prompt_log, f, indent=4)


def query_part_name(folder_path, part_folder, api_key, type_str=None):

    cache1_pth = f"{folder_path}/gpt4_query/segmentation.json"
    os.makedirs(os.path.split(cache1_pth)[0], exist_ok=True)

    # Initialize or load segmentation data
    if not os.path.exists(cache1_pth):
        with open(cache1_pth, 'w') as f:
            json.dump({}, f, indent=4)
    with open(cache1_pth) as f:
        data = json.load(f)

    # Iterate through parts in the folder
    for image in os.listdir(part_folder):
        name_of_part = image.split(".")[0]
        
        # Skip if part data already exists in the cache
        if name_of_part in data:
            print(f"Skipping {name_of_part}, data already exists.")
            continue

        # Initialize variables for retrying
        response, success, count = '', False, 0
        while count < 3 and not success:
            try:
                # Construct the prompt for querying
                new_prompt_shape = segmentation_prompt.format(type_str)
                # Suppress verbose output
                
                # Query the API
                image_path = os.path.join(part_folder, image)
                response = process_image(image_path, api_key, new_prompt_shape)
                
                # Check if there's an error in the response
                if 'error' in _get_response_json(response).keys():
                    print("Error detected, waiting for rate limit...")
                    time.sleep(5)
                    count += 1
                    continue
                # Success
                success = True
                # Suppress verbose output
                break
            except Exception as e:
                # Handle any unexpected exceptions and retry
                print(f"Error querying {name_of_part}: {e}")
                count += 1
                time.sleep(5)

        # Store the response if successful
        if success:
            data[name_of_part] = _get_response_json(response)
            with open(cache1_pth, 'w') as f:
                json.dump(data, f, indent=4)
        
    # Final return (optional, based on specific use case)
    return data


def query_segment_group(folder_path, part_folder, api_key, type_str=None):

    cache1_pth = f"{folder_path}/gpt4_query/segmentation.json"
    os.makedirs(os.path.split(cache1_pth)[0], exist_ok=True)

    # Initialize or load segmentation data
    if not os.path.exists(cache1_pth):
        with open(cache1_pth, 'w') as f:
            json.dump({}, f, indent=4)
    with open(cache1_pth) as f:
        data = json.load(f)

    # Iterate through parts in the folder
    for image in os.listdir(part_folder):

        name_of_part = image.split(".")[0]
        
        # Skip if part data already exists in the cache
        if name_of_part in data:
            print(f"Skipping {name_of_part}, data already exists.")
            continue

        # Initialize variables for retrying
        response, success, count = '', False, 0
        while count < 3 and not success:
            try:
                # Construct the prompt for querying
                new_prompt_shape = segmentation_prompt.format(type_str)
                # Suppress verbose output
                
                # Query the API
                image_path = os.path.join(part_folder, image)
                response = process_image(image_path, api_key, new_prompt_shape)
                
                # Check if there's an error in the response
                if 'error' in _get_response_json(response).keys():
                    print("Error detected, waiting for rate limit...")
                    time.sleep(5)
                    count += 1
                    continue
                # Success
                success = True
                # Suppress verbose output
                break
            except Exception as e:
                # Handle any unexpected exceptions and retry
                print(f"Error querying {name_of_part}: {e}")
                count += 1
                time.sleep(5)

        # Store the response if successful
        if success:
            data[name_of_part] = _get_response_json(response)
            with open(cache1_pth, 'w') as f:
                json.dump(data, f, indent=4)
        
    # Final return (optional, based on specific use case)
    return data



def query_main_type(folder_path, api_key, stitch_image_folder, type_str):
    
    # Define paths and ensure directories exist
    cache1_pth = f"{folder_path}/gpt4_query/segmentation.json"
    os.makedirs(os.path.split(cache1_pth)[0], exist_ok=True)
    cache2_pth = f"{folder_path}/gpt4_query/overall_query.json"

    previous_query = folder_path +  "/gpt4_query"
    segmentation_infor = os.path.join(previous_query, "segmentation.json")

    # Load segmentation data
    with open(segmentation_infor, 'r') as file:
        data = json.load(file)

    # Extract descriptions
    descriptions = {
        key: value["choices"][0]["message"]["content"] for key, value in data.items()
    }

    # Load existing results if they exist
    if os.path.exists(cache2_pth):
        with open(cache2_pth, 'r') as file:
            results = json.load(file)
    else:
        results = {}

    object_type = type_str

    for key, description in descriptions.items():
        # Skip API call if result for this key already exists
        if key in results:
            print(f"Skipping {key}, already processed.")
            continue
        
        # Use two separate images: full captured object + part mask
        captured_image_path = os.path.join(os.path.dirname(folder_path), "captured_images", "stitched_image.jpg")
        part_mask_path = os.path.join(stitch_image_folder, f"{key}.png")

        image_paths = [captured_image_path, part_mask_path]

        prompt_init = introduction_prompt.format(object_type, description)
        all_prompt = f"""Image 1 shows the full captured {object_type} object. Image 2 shows the segmented part mask for this specific part.

{prompt_init}

{prompt_overall}"""

        count = 0
        while count < 5:
            count += 1
            response = process_image_multi_qwen(image_paths, all_prompt, max_new_tokens=100, max_image_size=600)
            try:
                response_json = _get_response_json(response)
                desc = response_json["choices"][0]["message"]["content"]
                # Suppress verbose output

                if desc is not None:
                    c_mat = check_mat_valid(desc, count)
                    response_json["choices"][0]["message"]["content"] = c_mat
                    if c_mat is not None:
                        # Suppress verbose output
                        break
                
            except json.JSONDecodeError:
                print(f"Error decoding JSON for query {part_image}, retrying...")
                time.sleep(2)
                continue

        # Save response to results and update cache file
        results[key] = response_json
        with open(cache2_pth, 'w') as outfile:
            json.dump(results, outfile, indent=4)


def query_second_categories(folder_path, api_key, stitch_image_folder, object_type):

    # Define paths and ensure directories exist
    cache1_pth = f"{folder_path}/gpt4_query/segmentation.json"
    os.makedirs(os.path.split(cache1_pth)[0], exist_ok=True)
    cache2_pth = f"{folder_path}/gpt4_query/overall_query.json"
    previous_query = folder_path + "/gpt4_query"
    segmentation_infor = os.path.join(previous_query, "segmentation.json")
    cache3_pth = f"{folder_path}/gpt4_query/sub_cat.json"

    # Load segmentation data
    with open(segmentation_infor, 'r') as file:
        data = json.load(file)

    with open(cache2_pth, 'r') as file:
        materail_classifcation = json.load(file)

   
    # Extract descriptions
    descriptions = {
        key: value["choices"][0]["message"]["content"] for key, value in data.items()
    }

    materail_cat = {
        key: value["choices"][0]["message"]["content"] for key, value in materail_classifcation.items()
    }

    sub_lst = json.load(open("litereality_database/PBR_materials/material_lib/annotations/category_tree.json")) # category tree
    # new query

    # Load existing results if they exist
    if os.path.exists(cache3_pth):
        with open(cache3_pth, 'r') as file:
            results = json.load(file)
    else:
        results = {}

    for key, description in descriptions.items():
        # Skip API call if result for this key already exists
        if key in results:
            print(f"Skipping {key}, already processed.")
            continue

        # Use two separate images: full captured object + part mask
        captured_image_path = os.path.join(os.path.dirname(folder_path), "captured_images", "stitched_image.jpg")
        part_mask_path = os.path.join(stitch_image_folder, f"{key}.png")

        image_paths = [captured_image_path, part_mask_path]

        prompt_init = introduction_prompt.format(object_type, description)
        first_class_mat = materail_cat[key]

        sub_type = sub_lst[first_class_mat] # subtypes to be select from
        prompt = f"""Image 1 shows the full captured {object_type} object. Image 2 shows the segmented part mask for this specific part.

{prompt_init}

For the part highlighted in Image 2, analyze how it appears in the full object context of Image 1 and select the most similar {first_class_mat} material type (including color, pattern, roughness, age and so on...).

Available {first_class_mat} material types: {sub_type}

Output only a single word representing the category from the available list."""

        count = 0
        while count < 5:
            count += 1
            response = process_image_multi_qwen(image_paths, prompt, max_new_tokens=100, max_image_size=600)
            try:
                response_json = _get_response_json(response)
                desc = response_json["choices"][0]["message"]["content"]
                print("key", key)
                print(f"desc: {desc}")
                if desc is not None:
                    if desc in sub_lst[first_class_mat]:
                        # Suppress verbose output
                        break
                    elif desc == 'Stone':
                        response_json["choices"][0]["message"]["content"] = 'PavingStones'
                        # Suppress verbose output
                    else: continue
            except json.JSONDecodeError:
                print(f"Error decoding JSON for query {part_image}, retrying...")
                time.sleep(2)
                continue

        # Save response to results and update cache file
        results[key] = response_json
        with open(cache3_pth, 'w') as outfile:
            json.dump(results, outfile, indent=4)


def query_final_categories(folder_path, api_key, stitch_image_folder, object_type):

    # Define paths and ensure directories exist
    cache1_pth = f"{folder_path}/gpt4_query/segmentation.json"
    os.makedirs(os.path.split(cache1_pth)[0], exist_ok=True)
    cache2_pth = f"{folder_path}/gpt4_query/overall_query.json"
    cache3_pth = f"{folder_path}/gpt4_query/sub_cat.json"
    segmentation_infor = cache1_pth
    # Load segmentation data
    with open(segmentation_infor, 'r') as file:
        segmentation = json.load(file)

    with open(cache2_pth, 'r') as file:
        materail_classifcation = json.load(file)

    with open(cache3_pth, 'r') as file:
        sub_cat = json.load(file)


    # Extract descriptions
    descriptions = {
        key: value["choices"][0]["message"]["content"] for key, value in segmentation.items()
    }

    materail_cat = {
        key: value["choices"][0]["message"]["content"] for key, value in materail_classifcation.items()
    }

    sub_cat = {
        key: value["choices"][0]["message"]["content"] for key, value in sub_cat.items()   
    }

    sub_des = json.load(open("litereality_database/PBR_materials/material_lib/annotations/gpt_descriptions.json")) # highly-detailed annotation

    cache4_pth = f"{folder_path}/final_cat.json"
    # Load existing results if they exist
    if os.path.exists(cache4_pth):
        with open(cache4_pth, 'r') as file:
            results = json.load(file)
    else:
        results = {}

    for key, description in descriptions.items():
        # Skip API call if result for this key already exists
        if key in results:
            print(f"Skipping {key}, already processed.")
            continue
        # Use two separate images: full captured object + part mask
        captured_image_path = os.path.join(os.path.dirname(folder_path), "captured_images", "stitched_image.jpg")
        part_mask_path = os.path.join(stitch_image_folder, f"{key}.png")

        image_paths = [captured_image_path, part_mask_path]

        prompt_init = introduction_prompt.format(object_type, description)
        first_class_mat = materail_cat[key]
        second_class_mat = sub_cat[key]
        sub_type = sub_cat[key]
        full_description = sub_des[first_class_mat+'_'+sub_type] 
        prompt = f"""Image 1 shows the full captured {object_type} object. Image 2 shows the segmented part mask for this specific part.

{prompt_init}

{prompt_refine_sub_overall.format('', second_class_mat, full_description)}"""

        count = 0
        while count < 5:
            count += 1
            response = process_image_multi_qwen(image_paths, prompt, max_new_tokens=100, max_image_size=600)
            try:
                response_json = _get_response_json(response)
                desc = response_json["choices"][0]["message"]["content"]


                # desc = pp_dic_result(_get_response_json(response)["choices"][0]["message"]["content"])


                print("key", key)   
                print(f"desc: {desc}")
                

                if desc is not None:
                    # Suppress verbose output
                    break
                else:
                    continue
            except json.JSONDecodeError:
                print(f"Error decoding JSON for query {part_image}, retrying...")
                time.sleep(2)
                continue

        # Save response to results and update cache file
        results[key] = desc
        with open(cache4_pth, 'w') as outfile:
            json.dump(results, outfile, indent=4)


def query_top_10_materials(folder_path, api_key, stitch_image_folder, object_type, scene_path=None):
    """
    Phase 1: Text + Visual Pre-Selection to get top 10 materials per part.
    
    Uses VLM to analyze database descriptions + captured images together to select
    top 10 material candidates for each part (or fewer if less than 10 available).
    
    Args:
        folder_path: Path to folder containing query results
        api_key: Not used (kept for compatibility)
        stitch_image_folder: Path to stitched mask images
        object_type: Object type (e.g., "Chair")
        scene_path: Path to scene folder (for logging)
    
    Returns:
        dict: Top 10 materials per part (or fewer if less available)
    """
    from litereality.LR_mat_painting.utils.vlm_logger import log_vlm_conversation, prepare_image_info
    
    # Define paths
    cache1_pth = f"{folder_path}/gpt4_query/segmentation.json"
    cache2_pth = f"{folder_path}/gpt4_query/overall_query.json"
    cache3_pth = f"{folder_path}/gpt4_query/sub_cat.json"
    cache4_pth = f"{folder_path}/gpt4_query/top_10_materials.json"
    os.makedirs(os.path.split(cache4_pth)[0], exist_ok=True)

    # Load segmentation data
    with open(cache1_pth, 'r') as file:
        segmentation = json.load(file)
    
    with open(cache2_pth, 'r') as file:
        materail_classifcation = json.load(file)
    
    with open(cache3_pth, 'r') as file:
        sub_cat = json.load(file)
    
    # Extract descriptions
    descriptions = {
        key: value["choices"][0]["message"]["content"] for key, value in segmentation.items()
    }
    
    materail_cat = {
        key: value["choices"][0]["message"]["content"] for key, value in materail_classifcation.items()
    }
    
    sub_cat_dict = {
        key: value["choices"][0]["message"]["content"] for key, value in sub_cat.items()   
    }
    
    # Load material descriptions database
    descriptions_file = "litereality_database/PBR_materials/material_lib/annotations/gpt_descriptions.json"
    if os.path.exists(descriptions_file):
        sub_des = json.load(open(descriptions_file))
    else:
        print(f"Warning: Material descriptions file not found at {descriptions_file}")
        sub_des = {}
    
    # Helper functions for filesystem-based material retrieval (alternative approach)
    def get_actual_materials_for_category(subcategory_key):
        """Get actual material IDs that exist in the filesystem for a subcategory"""
        base_path = "litereality_database/PBR_materials/material_lib/pbr_maps/train"
        
        # Map subcategory key to main category (e.g., 'Metal_PaintedMetal' -> 'Metal')
        main_category = subcategory_key.split('_')[0]
        
        cat_path = os.path.join(base_path, main_category)
        if not os.path.exists(cat_path):
            return []
        
        materials = []
        for item in sorted(os.listdir(cat_path)):
            mat_path = os.path.join(cat_path, item)
            if os.path.isdir(mat_path) and os.path.exists(os.path.join(mat_path, 'basecolor.png')):
                materials.append(f'{main_category}_{item}')
        
        return materials[:20]  # Limit to 20 for VLM

    def generate_material_description(material_id):
        """Generate a description for a material based on its ID"""
        parts = material_id.split('_', 1)
        if len(parts) < 2:
            return f'A material'
        
        category = parts[0].lower()
        name = parts[1].lower().replace('_', ' ')
        
        # Clean up common prefixes
        name = name.replace('acg ', '').replace('acg_', '')
        
        if category == 'fabric':
            return f'A {name} fabric material'
        elif category == 'metal':
            return f'A {name} metal surface'
        elif category == 'wood':
            return f'A {name} wooden surface'
        elif category == 'ceramic':
            return f'A {name} ceramic material'
        elif category == 'concrete':
            return f'A {name} concrete surface'
        elif category == 'plastic':
            return f'A {name} plastic material'
        else:
            return f'A {name} material'
    
    # Load existing results if they exist
    if os.path.exists(cache4_pth):
        with open(cache4_pth, 'r') as file:
            results = json.load(file)
    else:
        results = {}
    
    # Get reference images
    reference_image_path = None
    if scene_path:
        reference_image_path = os.path.join(scene_path, "captured_images", "stitched_image.jpg")
    
    # Process each part
    for key, description in descriptions.items():
        # Skip if already processed
        if key in results:
            print(f"Skipping {key}, already processed.")
            continue
        
        prompt_init = introduction_prompt.format(object_type, description)
        first_class_mat = materail_cat[key]
        second_class_mat = sub_cat_dict[key]
        sub_type = sub_cat_dict[key]
        
        # Get all materials in this subcategory
        material_key = f"{first_class_mat}_{sub_type}"
        
        # Format material descriptions for prompt
        material_list_text = ""
        material_ids = []
        
        if material_key in sub_des:
            # Use descriptions from database
            material_descriptions = sub_des[material_key]
            for mat_desc_dict in material_descriptions:
                for mat_id, mat_desc in mat_desc_dict.items():
                    material_ids.append(mat_id)
                    material_list_text += f"- {mat_id}: {mat_desc}\n"
        else:
            # Fallback: Use filesystem-based approach
            print(f"Warning: Material key {material_key} not found in descriptions, using filesystem-based retrieval")
            material_ids = get_actual_materials_for_category(material_key)
            if not material_ids:
                print(f"Warning: No materials found for {material_key}, skipping part {key}")
                continue
            # Generate descriptions for materials found in filesystem
            for mat_id in material_ids:
                mat_desc = generate_material_description(mat_id)
                material_list_text += f"- {mat_id}: {mat_desc}\n"
        
        # Determine how many materials to select (target: 10, but use what's available if less)
        target_count = 10
        num_to_select = min(target_count, len(material_ids))
        
        if len(material_ids) < target_count:
            print(f"  [Info] Only {len(material_ids)} materials available for {key}, selecting all {len(material_ids)}")
        
        # Create prompt for top 10 selection (or fewer if less available)
        top_10_prompt = f"""{prompt_init}

From the detailed descriptions of {second_class_mat} materials below, select the TOP {num_to_select} most appropriate materials for this part:

{material_list_text}

CRITICAL: You MUST use the EXACT material IDs from the list above. Copy them exactly as shown (including underscores, capitalization, and numbers). Do NOT modify, generate, or invent new material IDs.

IMPORTANT CONTEXT: Most objects in this dataset are MODERN FURNITURE items photographed in contemporary settings. Unless the reference image clearly shows heavy rust, corrosion, or industrial wear, avoid selecting materials described as having "significant rusting", "extensive rust coverage", "deep rust-colored", "heavy corrosion", or similar heavily weathered appearances. Modern furniture typically uses clean, well-maintained materials.

Consider:
- Visual similarity to the reference image
- Material properties and appropriateness for {object_type} furniture
- OBJECT CONTEXT: This appears to be MODERN/CONTEMPORARY furniture - prefer CLEAN, WELL-MAINTAINED materials
- AVOID heavily weathered, rusted, or distressed materials unless the reference clearly shows similar damage
- For metal parts: Choose materials that look clean and functional, not industrial or heavily corroded
- For fabric/wood parts: Prefer materials that appear new and well-maintained
- Color and texture characteristics should match but prioritize appropriateness over minor texture similarities
- Overall style coherence with modern furniture aesthetics

Return format (JSON):
{{
  "top_{num_to_select}": ["material_id_1", "material_id_2", ..., "material_id_{num_to_select}"],
  "scores": {{"material_id_1": 0.95, "material_id_2": 0.92, ...}},
  "reasoning": "Brief explanation of selection criteria and why these materials were chosen"
}}

Select exactly {num_to_select} materials, ranked from best to worst match. Use ONLY the exact IDs from the list above."""
        
        # Use two separate images: full captured object + part mask
        captured_image_path = os.path.join(os.path.dirname(folder_path), "captured_images", "stitched_image.jpg")
        part_mask_path = os.path.join(stitch_image_folder, f"{key}.png")

        image_paths = [captured_image_path, part_mask_path]

        # Prepare input images for logging
        input_images = [
            prepare_image_info(captured_image_path, f"Reference image showing {object_type}", "reference_image"),
            prepare_image_info(part_mask_path, f"Part mask for {description}", "part_mask")
        ]

        # Update prompt to reference Image 1 and Image 2
        updated_prompt = f"""Image 1 shows the full captured {object_type} object. Image 2 shows the segmented part mask for this specific part.

{prompt_init}

From the detailed descriptions of {second_class_mat} materials below, select the TOP {num_to_select} most appropriate materials for the part shown in Image 2, considering how it appears in the full object context of Image 1:

{material_list_text}

CRITICAL: You MUST use the EXACT material IDs from the list above. Copy them exactly as shown (including underscores, capitalization, and numbers). Do NOT modify, generate, or invent new material IDs.

Consider:
- Visual similarity to the reference image (Image 1)
- Material properties that match the part's appearance in context
- How the material would look on this specific part of the object

Return your response in this exact JSON format:
{{
  "top_{num_to_select}": ["material_id_1", "material_id_2", ..., "material_id_{num_to_select}"],
  "scores": {{"material_id_1": 0.95, "material_id_2": 0.92, ...}},
  "reasoning": "Brief explanation of selection criteria and why these materials were chosen"
}}

Select exactly {num_to_select} materials, ranked from best to worst match. Use ONLY the exact IDs from the list above."""

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
                # Use token limit for top 10 selection (needs ~1200 tokens for JSON with 10 IDs + scores + reasoning)
                response = process_image_multi_qwen(image_paths, updated_prompt, max_new_tokens=1500, max_image_size=800)
                processing_time = time.time() - start_time
                
                # process_image_qwen already returns OpenAI-compatible format
                response_json = _get_response_json(response)
                raw_response = response_json
                
                # Parse response
                content = response_json["choices"][0]["message"]["content"]

                # Try to extract JSON from response - handle multiple formats
                import re
                parsed_result = None
                
                # Try 1: Extract JSON from markdown code blocks
                json_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', content)
                if json_match:
                    try:
                        json_str = json_match.group(1)
                        parsed_result = json.loads(json_str)
                    except json.JSONDecodeError:
                        pass
                
                # Try 2: Extract JSON object directly
                if not parsed_result:
                    json_match = re.search(r'\{[\s\S]*\}', content)
                    if json_match:
                        try:
                            json_str = json_match.group(0)
                            parsed_result = json.loads(json_str)
                        except json.JSONDecodeError:
                            pass
                
                # Try 3: Look for top_10 or top_20 array even if JSON structure is imperfect
                if not parsed_result:
                    # Try to find material IDs in the response
                    material_id_pattern = r'["\']?([A-Za-z]+_[A-Za-z0-9_]+)["\']?'
                    found_ids = re.findall(material_id_pattern, content)
                    # Filter to only valid material IDs (must be in material_ids list)
                    valid_ids = [mid for mid in found_ids if mid in material_ids]
                    if len(valid_ids) >= num_to_select:
                        # Take first num_to_select unique
                        unique_ids = []
                        seen = set()
                        for mid in valid_ids:
                            if mid not in seen:
                                unique_ids.append(mid)
                                seen.add(mid)
                            if len(unique_ids) >= num_to_select:
                                break
                        if len(unique_ids) == num_to_select:
                            parsed_result = {
                                "top_10": unique_ids,  # Use top_10 key for consistency
                                "scores": {mat_id: 0.9 - (i * 0.01) for i, mat_id in enumerate(unique_ids)},
                                "reasoning": "Extracted from response text (JSON parsing failed)"
                            }
                
                # Validate structure - check for any top_N field (flexible parsing)
                if parsed_result:
                    # Check for any top_N field (top_10, top_20, top_8, etc.)
                    materials_list = None
                    top_key = None

                    # Look for any key that starts with "top_"
                    for result_key in parsed_result.keys():
                        if result_key.startswith("top_"):
                            materials_list = parsed_result[result_key]
                            top_key = result_key
                            break

                    if materials_list and isinstance(materials_list, list):
                        # Be flexible about the count - accept as long as we have a reasonable number
                        if len(materials_list) >= 4:  # Accept at least 4 materials
                            # Create normalized lookup (case-insensitive, strip whitespace)
                            material_ids_normalized = {mid.lower().strip(): mid for mid in material_ids}

                            # Validate all material IDs exist (with normalization)
                            valid_materials = []
                            invalid_materials = []

                            for returned_id in materials_list:
                                normalized = returned_id.strip()
                                # Try exact match first
                                if normalized in material_ids:
                                    valid_materials.append(normalized)
                                # Try case-insensitive match
                                elif normalized.lower() in material_ids_normalized:
                                    valid_materials.append(material_ids_normalized[normalized.lower()])
                                else:
                                    invalid_materials.append(returned_id)

                            # Accept partial results if we have enough valid materials
                            min_required = max(4, num_to_select // 2)  # At least 4 for top 10, or half for other cases
                            
                            if len(valid_materials) >= min_required:
                                # Fill missing slots with valid alternatives if needed
                                if len(valid_materials) < num_to_select:
                                    # Get remaining valid materials that weren't selected
                                    remaining_valid = [mid for mid in material_ids if mid not in valid_materials]
                                    # Fill up to num_to_select
                                    needed = num_to_select - len(valid_materials)
                                    valid_materials.extend(remaining_valid[:needed])
                                    
                                    if len(valid_materials) < num_to_select:
                                        print(f"  ⚠️ Warning: Only {len(valid_materials)}/{num_to_select} materials available for {key}")
                                
                                # Update parsed_result with normalized IDs and use top_10 key
                                parsed_result["top_10"] = valid_materials[:num_to_select]  # Take only what we need
                                if "top_20" in parsed_result:
                                    del parsed_result["top_20"]  # Remove old key
                                
                                if len(invalid_materials) > 0:
                                    print(f"  ⚠️ Warning: Replaced {len(invalid_materials)} invalid material(s) for {key}: {invalid_materials[:3]}")
                                
                                success = True
                                break
                            else:
                                # Debug output
                                if count == 1:
                                    print(f"  [Debug] Invalid IDs for {key}: {invalid_materials[:5]}")
                                    print(f"  [Debug] Sample valid IDs: {material_ids[:5]}")
                                parsing_errors.append(f"Found {len(valid_materials)}/{num_to_select} valid material IDs (minimum {min_required} required). Invalid: {invalid_materials[:3]}")
                        else:
                            parsing_errors.append(f"Expected {num_to_select} materials, got {len(materials_list)}")
                    else:
                        parsing_errors.append("Missing 'top_N' field in response (expected top_10, top_20, etc.)")
                else:
                    parsing_errors.append("No JSON found in response")
                
            except Exception as e:
                parsing_errors.append(f"Error on attempt {count}: {str(e)}")
                time.sleep(2)
                continue
        
        # Log conversation
        if scene_path:
            object_name_clean = ''.join(filter(lambda x: not x.isdigit(), object_type))
            log_vlm_conversation(
                scene_path=scene_path,
                phase="phase_1_text_visual_selection",
                part_id=key,
                object_name=object_name_clean,
                prompt=top_10_prompt,
                input_images=input_images,
                raw_response=raw_response or {},
                parsed_result=parsed_result,
                metadata={
                    "material_database_filter": material_key,
                    "num_candidates_considered": len(material_ids),
                    "vlm_model_version": "qwen3-vl"
                },
                prompt_template="top_20_selection",
                parsing_errors=parsing_errors,
                success=success,
                error_message=None if success else "Failed to parse top 20 materials"
            )
        
        # Save result - NO FALLBACK, fail loudly if parsing failed
        if success and parsed_result:
            results[key] = {
                **parsed_result,
                "prompt": updated_prompt  # Include the prompt used for transparency
            }
            print(f"  ✓ Successfully parsed top {num_to_select} materials for {key}")
        else:
            # CRITICAL ERROR: Parsing failed after all retries
            content_preview = ""
            if raw_response and "choices" in raw_response and len(raw_response["choices"]) > 0:
                content_preview = raw_response["choices"][0]["message"]["content"][:1000]
            
            error_msg = f"❌ CRITICAL: Failed to parse top {num_to_select} materials for {key} after 5 attempts!"
            error_msg += f"\n   Parsing errors: {parsing_errors}"
            error_msg += f"\n   Response content length: {len(content_preview)} chars"
            error_msg += f"\n   Response preview:\n{content_preview}"
            print(error_msg)
            raise ValueError(
                f"Failed to parse VLM response for part {key}. "
                f"This is a critical error - material selection cannot proceed. "
                f"Check the VLM conversation logs for details. "
                f"Parsing errors: {parsing_errors}\n"
                f"Response preview: {content_preview[:500]}"
            )
        
        # Save to cache
        with open(cache4_pth, 'w') as outfile:
            json.dump(results, outfile, indent=4)
    
    return results


def format_string(s):
    """
    Formats a given string to help find correct name from material lib.
    
    Args:
    s (str): The input string to be formatted.

    Returns:
    str: The formatted string.

    Example:
    >>> input_str = 'MetalPlates015A'
    >>> formatted_str = format_string(input_str)
    >>> print(formatted_str)
    'acg_metal_plates_015_a'
    """
    s = re.sub(r'(?<!^)(?=[A-Z])', '_', s)
    s = re.sub(r'(\d)([A-Za-z])', r'\1_\2', s)
    s = re.sub(r'([A-Za-z])(\d)', r'\1_\2', s)
    return 'acg_' + s.lower()


import ast

def link_pbr_materials(folder_path, mkdir_select_mat):
    final_cat_path = os.path.abspath(f"{folder_path}/final_cat.json")
    with open(final_cat_path, 'r') as file:
        final_cat = json.load(file)

    mkdir_select_mat = os.path.abspath(mkdir_select_mat)
    os.makedirs(mkdir_select_mat, exist_ok=True)

    for key, value in final_cat.items():
        # Parse JSON
        cleaned_value = value
        material_dict = ast.literal_eval(cleaned_value)

        try:
            data_dict = material_dict
            select_mat = list(data_dict.keys())[0]
            first_layer = select_mat.split("_")[0]
            full_name = select_mat.split("_")[1]
            path = os.path.abspath(f"litereality_database/PBR_materials/material_lib/pbr_maps/train/{first_layer}/{format_string(full_name)}/")
            # Suppress verbose output
        except json.JSONDecodeError:
            # Suppress verbose output
            continue
        
        # Create or replace symbolic link
        target_link = f"{mkdir_select_mat}/{key}"
        if os.path.islink(target_link) or os.path.exists(target_link):
            os.remove(target_link)  # Remove existing link or file
            # Suppress verbose output
        
        try:
            os.symlink(path, target_link)
            # Suppress verbose output
        except OSError as e:
            print(f"Error creating symlink: {e}")
        

def reduce_materail_size(scene_name, reduce_size):

    object_list = os.listdir(f"output/Onboarded/{scene_name}")

    for object_name in object_list:

        print(f"Resizing images for {object_name} in {scene_name}...")

        # Define the source and destination folders
        source_folder = f"output/Onboarded/{scene_name}/{object_name}/decomposed/select_mat"
        destination_folder = f"output/Onboarded/{scene_name}/{object_name}/decomposed/select_mat_reduced"

        new_size = (reduce_size, reduce_size)  # Target resolution for resizing

        os.makedirs(destination_folder, exist_ok=True)

        for root, dirs, files in os.walk(source_folder, followlinks=True):
            print(f"Processing directory: {root}")
            
            for file in files:
                if file.endswith(".png"):
                    # Define source and destination paths
                    source_path = os.path.join(root, file)
                    
                    # Replicate folder structure in destination folder
                    relative_path = os.path.relpath(root, source_folder)
                    dest_dir = os.path.join(destination_folder, relative_path)
                    os.makedirs(dest_dir, exist_ok=True)
                    destination_path = os.path.join(dest_dir, file)
                    
                    try:
                        # Open the image
                        with Image.open(source_path) as img:
                            # Store the original mode
                            original_mode = img.mode

                            # Convert to RGB if not in RGB mode
                            if original_mode not in ("RGB", "RGBA"):
                                img = img.convert("RGB")
                            
                            # Resize the image
                            resized_img = img.resize(new_size, Image.LANCZOS)

                            # Convert back to the original mode, if necessary
                            if original_mode != "RGB":
                                resized_img = resized_img.convert(original_mode)
                            
                            # Save the resized image in the destination path
                            resized_img.save(destination_path)
                            print(f"Saved resized image to {destination_path}")

                    except Exception as e:
                        print(f"Error processing {source_path}: {e}")



query_cropping = """

This image has two sections. The top section is a real-life photo of a {}. Several black bounding boxes indicate cropped material areas in this image, each labeled with a white index on a black background. Below the photo is a 3D rendering of a retrieved model. The focus area is highlighted in red, which was described as {}. Although the 3D model does not exactly match the furniture in the image, it closely resembles it. The task is to find a material for the red-highlighted part, so the retrieved model can be rendered to resemble the real-life item in the top image. Using both visual information and rendering knowledge, please identify if any of the cropped material areas correspond to the red-highlighted part. If a match exists, directly output the index number of the corresponding patch. In some cases, the segmented part might not be present in the top image but may have a similar material to other parts. This is acceptable, do your best to give me a prediction. If finding a match is too difficult, simply output None. Direct output the index number or None without any other words or punctuation.

"""

def query_crop_selection(folder_path, cropped_image_folder, api_key, type_str=None):

    previous_query = folder_path +  "/gpt4_query"
    segmentation_infor = os.path.join(previous_query, "segmentation.json")
    with open(segmentation_infor, 'r') as file:
        segmentation_info = json.load(file)

    # Extract descriptions
    descriptions = {
        key: value["choices"][0]["message"]["content"] for key, value in segmentation_info.items()
    }

    # check if the folder exists
    if not os.path.exists(cropped_image_folder):
        print(f"Warning: Cropped image folder does not exist: {cropped_image_folder}")
        return

    all_frames = os.listdir(cropped_image_folder)

    # Create a log file for prompts
    prompt_log_file = f"{folder_path}/gpt4_query/crop_selection_prompts.json"
    prompt_log = {}
    if os.path.exists(prompt_log_file):
        with open(prompt_log_file) as f:
            prompt_log = json.load(f)

    for frame in all_frames:    
        crop_selection = f"{folder_path}/gpt4_query/crop_selection_{frame}.json"
        os.makedirs(os.path.split(crop_selection)[0], exist_ok=True)

        if not os.path.exists(crop_selection):
            with open(crop_selection, 'w') as f:
                json.dump({}, f, indent=4)
        with open(crop_selection) as f:
            crop_selection_dict = json.load(f)

        for name_of_part, description in descriptions.items():
            image_path = os.path.join(cropped_image_folder, frame, f"combined_with_{name_of_part}.png")
            prompt_crop = query_cropping.format(type_str, description)
            
            # Save prompt to log (frame is already in format "frame_XX_output")
            frame_key = frame  # frame is already "frame_XX_output"
            if frame_key not in prompt_log:
                prompt_log[frame_key] = {}
            prompt_log[frame_key][name_of_part] = {
                "prompt": prompt_crop,
                "part_description": description,
                "input_image": image_path
            }
            
            if name_of_part in crop_selection_dict:
                continue
            response, success, count = '', False, 0
            while count < 3 and not success:
                try:
                    # Query the API
                    response = process_image(image_path, api_key, prompt_crop)
                    # Check if there's an error in the response
                    if 'error' in _get_response_json(response).keys():
                        print("Error detected, waiting for rate limit...")
                        time.sleep(5)
                        count += 1
                        continue
                    # Success
                    success = True
                    # Suppress verbose output
                    break
                except Exception as e:
                    # Handle any unexpected exceptions and retry
                    print(f"Error querying {name_of_part}: {e}")
                    count += 1
                    time.sleep(5)

            # Store the response if successful
            if success:
                crop_selection_dict[name_of_part] = _get_response_json(response)["choices"][0]["message"]["content"]
                # Suppress verbose output

                with open(crop_selection, 'w') as f:
                    json.dump(crop_selection_dict, f, indent=4)
    
    # Save prompt log
    with open(prompt_log_file, 'w') as f:
        json.dump(prompt_log, f, indent=4)

import glob 
import shutil

def extract_cropped_patches(scene):
    # Base directory for GPT output and cropped images
    gpt_output_dir = f"{scene}"
    cropped_images_dir = f"{scene}/cropped_image_all/"
    
    # Iterate through objects in the GPT output directory
    for object_name in os.listdir(gpt_output_dir):
        object_dir = os.path.join(gpt_output_dir, object_name)
        gpt_query_dir = os.path.join(object_dir, "gpt4_query")
        output_path = f"{scene}/select_crop/{object_name}"

        # Ensure the gpt_query directory exists for the current object
        if not os.path.isdir(gpt_query_dir):
            continue

        # Find all crop selection JSON files in the gpt_query directory
        crop_selection_files = glob.glob(os.path.join(gpt_query_dir, "crop_selection_frame_*_output.json"))

        # Process each JSON file to retrieve cropping information
        for crop_selection_file in crop_selection_files:
            # Extract the frame number from the filename
            frame_number = crop_selection_file.split("_")[-2]

            # Load JSON data from the crop selection file
            with open(crop_selection_file, 'r') as f:
                crop_data = json.load(f)

            # Create directories and copy cropped images based on JSON data
            for crop_key, crop_index in crop_data.items():
                # Construct the folder path for this object part
                target_folder = os.path.join(output_path, crop_key)
                os.makedirs(target_folder, exist_ok=True)

                print("target_folder", target_folder)
                object_name_withoutgpt = object_name.split("_")[0]
                # Path to the cropped image based on crop index
                crop_image_path = os.path.join(cropped_images_dir, object_name_withoutgpt, f"frame_{frame_number}_output", f"crop_{crop_index}.jpg")

                # Check if the crop image exists before copying
                if os.path.exists(crop_image_path):
                    shutil.copy(crop_image_path, target_folder)
                else:
                    print(f"Warning: Cropped image {crop_image_path} does not exist.")

def extract_holistic_cropped_patches(scene):
    """
    Extract and save crop patches based on holistic matching results.
    This function takes the JSON results from holistic matching and 
    copies the relevant crop images to target folders.
    
    Args:
        scene: Path to the scene directory
    """
    # Setup paths
    object_name = scene.split("/")[-1]
    # Remove digits from object name
    object_type = object_name.translate({ord(i): None for i in '0123456789'})
    
    # Base directory paths
    folder_path = f"{scene}/{object_type}_gpt"
    gpt_query_dir = os.path.join(folder_path, "gpt4_query")
    cropped_images_dir = f"{scene}/cropped_image_all/"
    
    # Find all holistic matching JSON files
    holistic_matching_files = glob.glob(os.path.join(gpt_query_dir, "holistic_matching_frame_*_output.json"))
    
    if not holistic_matching_files:
        print(f"No holistic matching files found in {gpt_query_dir}")
        return
    
    print(f"Found {len(holistic_matching_files)} holistic matching files")
    
    # Process each JSON file to retrieve cropping information
    for matching_file in holistic_matching_files:
        # Extract the frame number from the filename
        frame_number = matching_file.split("_")[-2]
        
        # Load JSON data from the holistic matching file
        with open(matching_file, 'r') as f:
            matching_data = json.load(f)
        
        # Create directories and copy cropped images based on JSON data
        for part_name, part_info in matching_data.items():
            # Get the crop number from the matching data
            crop_index = part_info.get("best_crop_match")
            
            # Skip if there's no good match
            if crop_index == "no_match":
                print(f"No match found for {part_name} in frame {frame_number}")
                continue
                
            # Construct the target folder path for this object part
            target_folder = os.path.join(scene, "select_crop", f"{object_type}_gpt", part_name)
            os.makedirs(target_folder, exist_ok=True)
            
            print(f"Saving crop for {part_name} to {target_folder}")
            
            # Path to the cropped image based on crop index
            crop_image_path = os.path.join(cropped_images_dir, object_type, f"frame_{frame_number}_output", f"crop_{crop_index}.jpg")
            
            # Check if the crop image exists before copying
            if os.path.exists(crop_image_path):
                # Copy the cropped image to the target folder
                shutil.copy(crop_image_path, target_folder)
                # Suppress verbose output
            else:
                print(f"Warning: Cropped image {crop_image_path} does not exist.")

def query_crop_selection_holistic(folder_path, cropped_image_folder, api_key, type_str=None):
    """
    Process holistic crop selection that matches segments to crops all at once
    instead of individually processing each part.
    
    Args:
        folder_path: Path to the folder containing the GPT-4 query results
        cropped_image_folder: Path to the folder containing the cropped images
        api_key: OpenAI API key
        type_str: Type of the object (e.g., Chair, Table)
    
    Returns:
        Dictionary of matching results for each frame
    """
    previous_query = folder_path + "/gpt4_query"
    segmentation_infor = os.path.join(previous_query, "segmentation.json")
    
    # Load segmentation data
    with open(segmentation_infor, 'r') as file:
        segmentation_info = json.load(file)

    # Extract descriptions
    descriptions = {
        key: value["choices"][0]["message"]["content"] for key, value in segmentation_info.items()
    }
    
    # Check if the folder exists
    if not os.path.exists(cropped_image_folder):
        print(f"Warning: Cropped image folder does not exist: {cropped_image_folder}")
        return
    
    # Identify frames
    all_frames = os.listdir(cropped_image_folder)
    
    results = {}
    
    for frame in all_frames:
        # Path to the holistic image with all segments annotated
        holistic_image_path = os.path.join(cropped_image_folder, frame, "annotated_with_segments.png")
        
        if not os.path.exists(holistic_image_path):
            print(f"Warning: Holistic image does not exist: {holistic_image_path}")
            continue
            
        # Create output path for the holistic matching results
        holistic_results_path = f"{folder_path}/gpt4_query/holistic_matching_{frame}.json"
        os.makedirs(os.path.split(holistic_results_path)[0], exist_ok=True)
        
        # Skip if we've already processed this frame
        if os.path.exists(holistic_results_path):
            print(f"Skipping {frame}, already processed.")
            with open(holistic_results_path, 'r') as f:
                results[frame] = json.load(f)
            continue
        
        # Construct the holistic prompt with all segment descriptions
        parts_descriptions = ""
        for part_name, description in descriptions.items():
            parts_descriptions += f"- {part_name}: {description}\n"
            
        holistic_prompt = f"""
You are given an image with two parts:
- Top part: A real-world image showing a {type_str} with numbered rectangular crops (1 to N).
- Bottom part: Rendered images of segmented 3D object models. Each segment is highlighted in red in different images.

Here are the descriptions of each segmented part:
{parts_descriptions}

Your task is to match each segmented part to the most visually similar crop from the top image.

For each part:
1. Consider the segmented part based on the image and its text description.
2. Compare it against the top image and determine which crop best shows the same visual part.
3. Decide the best crop or say "no good match."

Output Format (JSON format):
{{
  "part_name_1": {{
    "description": "brief description of the part",
    "best_crop_match": "crop number or no_match",
    "reason": "explanation for the match"
  }},
  "part_name_2": {{
    "description": "brief description of the part",
    "best_crop_match": "crop number or no_match",
    "reason": "explanation for the match"
  }},
  ...
}}

Choose the most complete and visible crop for each part. If a segment is not visible or has no good match, set "best_crop_match" to "no_match".
IMPORTANT: Please provide a valid JSON response only. Ensure all property names and values are enclosed in double quotes and follow proper JSON syntax.
"""

        # Save prompt to log
        prompt_log_file = f"{folder_path}/gpt4_query/crop_selection_prompts.json"
        prompt_log = {}
        if os.path.exists(prompt_log_file):
            with open(prompt_log_file) as f:
                prompt_log = json.load(f)
        
        # frame is already in format "frame_XX_output"
        frame_key = f"{frame}_holistic"
        prompt_log[frame_key] = {
            "prompt": holistic_prompt,
            "input_image": holistic_image_path,
            "part_descriptions": parts_descriptions
        }
        
        with open(prompt_log_file, 'w') as f:
            json.dump(prompt_log, f, indent=4)

        # Process the holistic image
        response, success, count = '', False, 0
        while count < 3 and not success:
            try:
                # Query the API
                response = process_image(holistic_image_path, api_key, holistic_prompt)
                # Check if there's an error in the response
                if 'error' in _get_response_json(response).keys():
                    print("Error detected, waiting for rate limit...")
                    time.sleep(5)
                    count += 1
                    continue
                # Success
                success = True
                # Suppress verbose output
                break
            except Exception as e:
                # Handle any unexpected exceptions and retry
                print(f"Error querying holistic matching for {frame}: {e}")
                count += 1
                time.sleep(5)

        # Store the response if successful
        if success:
            try:
                # Parse the JSON response from the model
                response_json = _get_response_json(response)
                content = response_json["choices"][0]["message"]["content"]
                # Suppress verbose output
                
                # Extract JSON from the response (it might be wrapped in markdown code blocks)
                json_content = content
                if "```json" in content:
                    json_content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    json_content = content.split("```")[1].split("```")[0].strip()
                
                # Fix common JSON formatting errors
                try:
                    # First attempt to parse as-is
                    matching_results = json.loads(json_content)
                except json.JSONDecodeError:
                    # Clean up the JSON content silently
                    # Replace single quotes with double quotes
                    json_content = json_content.replace("'", '"')
                    # Fix property names not in quotes
                    import re
                    json_content = re.sub(r'(\s*)([a-zA-Z0-9_]+)(\s*):([^:])', r'\1"\2"\3:\4', json_content)
                    # Fix trailing commas
                    json_content = re.sub(r',(\s*[}\]])', r'\1', json_content)
                    # Fix missing commas between objects
                    json_content = re.sub(r'}\s*{', '},{', json_content)
                    
                    try:
                        # Try parsing the cleaned JSON
                        matching_results = json.loads(json_content)
                    except json.JSONDecodeError:
                        # Save the problematic content for debugging (silently)
                        raw_content_path = f"{folder_path}/gpt4_query/raw_response_{frame}.txt"
                        with open(raw_content_path, 'w') as f:
                            f.write(content)
                        
                        # Fallback: try to extract the most important information using regex
                        try:
                            # Extract part names and best crop matches using regex
                            part_matches = re.findall(r'"([^"]+)"[^{]*{[^}]*"best_crop_match"[^:]*:[^"]*"([^"]+)"', content)
                            
                            if part_matches:
                                matching_results = {}
                                for part_name, crop_match in part_matches:
                                    matching_results[part_name] = {
                                        "description": "Auto-extracted", 
                                        "best_crop_match": crop_match,
                                        "reason": "Extracted from malformed JSON"
                                    }
                                # Suppress verbose output
                            else:
                                continue
                        except Exception as extract_error:
                            continue
                
                # Save the successfully parsed results
                with open(holistic_results_path, 'w') as f:
                    json.dump(matching_results, f, indent=4)
                    
                results[frame] = matching_results
                # Suppress verbose output
                
            except Exception as e:
                print(f"Error processing response for {frame}: {e}")
                # Try to save whatever we can for debugging
                try:
                    with open(f"{folder_path}/gpt4_query/error_response_{frame}.txt", 'w') as f:
                        f.write(str(response_json))
                except:
                    print("Could not save error response")
                
    return results