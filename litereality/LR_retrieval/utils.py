import os, json, time
import base64
import re

# Import Qwen3-VL functions
try:
    from qwan import load_model as qwen_load_model, inference as qwen_inference, inference_multi as qwen_inference_multi
    QWEN_AVAILABLE = True
except ImportError:
    QWEN_AVAILABLE = False
    print("Warning: Qwen3-VL not available. Using OpenAI API.")

# Import subcategory queries
try:
    from litereality.LR_mat_painting.utils.gpt_query_summary import retrival_query, query_sub_class
    SUBQUERY_AVAILABLE = True
except ImportError:
    SUBQUERY_AVAILABLE = False
    print("Warning: gpt_query_summary not available. Two-stage retrieval will use fallback prompts.")


# ------------------ prompt section ------------------

prompt_for_categories = """
I will provide an image composed of four sub-images arranged in a 2x2 grid. Each sub-image contains a red bounding box highlighting the {} object from different viewpoints.
Carefully examine all four images to understand the object's shape, appearance, and details. Then, based on your best judgment, determine which category from my database the highlighted object most likely belongs to.

IMPORTANT: You must select the MOST SPECIFIC (leaf-level) category from the database. Do NOT select parent categories like "Chair", "Bed", "Sofa", "Table", "Storage", "Others", etc. Instead, select a specific subcategory such as "Dining_Chair", "Double_Bed", "L-shaped_Sofa", "Coffee_Tea_Tables", etc.

The following is the object category structure in JSON dictionary format:{}

Output only the most specific category type text string from the provided database list—no explanations. This should be just text, no JSON formatting is needed.

"""

# Two-stage category retrieval prompts
# Note: prompt_stage1_parent_category is now built dynamically from JSON data
# This template will be formatted with semantic and category_list
prompt_stage1_parent_category_template = """
I will provide an image composed of four sub-images arranged in a 2x2 grid. Each sub-image contains a red bounding box highlighting the {OBJECT_CATEGORY} object from different viewpoints.

Examine all four images and confirm the main parent category of this {OBJECT_CATEGORY} object. Choose ONE of the following parent categories:
{category_list}

Output only the parent category name exactly as listed above—no explanations, no additional text.
"""

prompt_stage2_description = """
I will provide an image composed of four sub-images arranged in a 2x2 grid. Each sub-image contains a red bounding box highlighting a {} object from different viewpoints.

First, carefully examine all four images and provide a detailed description of this {} object. Focus on:
- Shape and structure (e.g., height, width, overall form)
- Key distinguishing features (e.g., backrest style, number of seats, leg design)
- Functional characteristics (e.g., foldable, adjustable, fixed)
- Any other notable geometric or structural features

Provide a clear, detailed description of the object's physical characteristics.
"""

prompt_stage2_subcategory_match = """
Based on the following description of a {} object:

{}

Now, examine the image again and determine which of these {} subcategories best matches the described object:

{}

Output only the category number and its name in the format "number. Category_Name" (e.g., "2. Dining_Chair")—no additional explanations.
"""


Prompt_final_select =  """
1.	You are given a single large image that contains two sections:
	•	Left Section: Four real RGB images of an object, each with a bounding box highlighting the object.
	•	Right Section: Four 3D templates labeled 1, 2, 3, and 4.
	2.	Ignore color, as the object's textures will be repainted later.
	3.	Analyze the left section to understand the object's shape, structure, and style, but not color.
	4.	Compare these characteristics to those of the four 3D templates in the right section.
	5.	If the object has a non-symmetric structure, ensure that its overall orientation matches when selecting the best template.
	6.	Select exactly one template whose overall design best matches the real object.
	7.	Output only the single number (1, 2, 3, or 4) corresponding to your final choice, with no extra text."""

Prompt_final_select_multi = """
You are given 5 images:
- Image 1: A stitched image showing four real RGB images of an object from different viewpoints, each with a bounding box highlighting the object. This is the reference object you need to match.
- Images 2-5: Four 3D template candidates labeled 1, 2, 3, and 4 respectively. These are the options you need to choose from.

IMPORTANT: Ignore color, texture, and material completely. Focus ONLY on geometric shape, structure, and form. Color and materials will be repainted later. Note that objects may be scaled, so absolute dimensions are not important - focus on relative proportions and geometric structure.

Your task is to find the template whose GEOMETRY best matches the reference object. Follow this reasoning process:

1. GEOMETRIC ANALYSIS of Image 1 (reference):
   - Identify key structural elements (e.g., legs, arms, backrest, seat for chairs; tabletop, base for tables)
   - Note relative proportions and component relationships (how components relate to each other in size and position)
   - Observe curvature, angles, and overall silhouette/profile
   - Check for symmetry or asymmetry
   - Note any distinctive geometric features

2. COMPARISON with each candidate (Images 2-5):
   For each template (1, 2, 3, 4), compare:
   - Do the structural elements match? (same number and arrangement)
   - Do the relative proportions match? (how components relate to each other, not absolute sizes)
   - Does the overall silhouette/profile match?
   - Do the geometric details match? (curvature, angles, edges)
   - Does the orientation match if the object is asymmetric?

3. SELECTION:
   - Eliminate candidates that clearly don't match the geometric structure
   - Among remaining candidates, choose the one with the closest geometric match
   - Prioritize structural similarity over minor details

Output ONLY the single number (1, 2, 3, or 4) corresponding to your final choice, with no explanations or extra text."""

Prompt_reference_description = """
You are given Image 1: A stitched image showing four real RGB images of an object from different viewpoints, each with a bounding box highlighting the object.

TASK: Describe the GEOMETRIC STRUCTURE of this object. Completely ignore color, texture, material, and surface appearance.

Describe the following geometric aspects:

1. STRUCTURAL ELEMENTS: What are the main components/parts of this object? (e.g., base, top, supports, arms, back, legs, etc.)
2. OVERALL SHAPE: What is the general form? (rectangular, circular, curved, angular, organic, etc.)
3. PROPORTIONS: How do the components relate to each other in size and position? (e.g., wide base, tall supports, etc.)
4. GEOMETRIC DETAILS: Any distinctive features? (curves, angles, edges, symmetry, asymmetry, etc.)
5. PROFILE/SILHOUETTE: What does the overall outline look like from different viewpoints?

Be specific and focus ONLY on geometry and structure. Do not mention color, material, or texture.
"""

Prompt_verification_visual = """
You are given 5 images:
- Image 1: A stitched image showing four real RGB images of an object from different viewpoints (the reference object)
- Images 2-5: Four 3D template candidates labeled 1, 2, 3, and 4 respectively

A template (X) was previously selected as the best geometric match for Image 1.

VERIFICATION TASK: Visually examine Image 1 and Template X side by side.

Check if Template X matches Image 1 in:
- Structural elements: Same components/parts in same arrangement
- Overall shape and form: Similar geometric structure
- Proportions: Components relate to each other similarly
- Geometric details: Matching curves, angles, edges, symmetry

IMPORTANT: Ignore color, texture, and material completely. Focus ONLY on geometry.

If Template X is a good geometric match, output "X" (the number).
If Template X is NOT a good match, output "wrong".

Output ONLY: X or "wrong".
"""


def process_image(image_path, api_key, prompt):
    # Function to encode the image
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
        
    # Getting the base64 string
    base64_image = encode_image(image_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4o",
        "messages": [
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": prompt
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
                }
            ]
            }
        ],
        "max_tokens": 300
    }
    api_base = "https://api.openai.com/v1/chat/completions" # you can switch to another accessible GPT-4 API host
    # response = requests.post(api_base, headers=headers, json=payload) # another way to get response
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(**payload)
    return response

import os
import json
import time

def multi_try_ensure_response(image_path, api_key, prompt, save_folder, max_tries=5, Force = False):
    save_json_path = os.path.join(save_folder, "response.json")
    save_txt_path = os.path.join(save_folder, "response.txt")

    if not Force:
        # Check if response.json already exists and is not empty
        if os.path.exists(save_json_path) and os.path.getsize(save_json_path) > 0:
            with open(save_json_path, "r") as f:
                try:
                    response = json.load(f)
                    message = response['choices'][0]['message']['content']
                    print("output response found:", message)

                    # Save message to text file
                    with open(save_txt_path, "w") as txt_file:
                        txt_file.write(message)
                    print(f"Message saved to: {save_txt_path}")

                    return message
                except (json.JSONDecodeError, KeyError, IndexError):
                    print("Invalid or corrupted JSON file, proceeding with new request...")

    response, count, success = None, 0, False
    while count < max_tries:
        try:
            response = process_image(image_path, api_key, prompt)  # Assuming this is a function you have
            json_response = json.loads(response.json())
            if 'error' in json_response:
                print("Waiting for rate limit...")
                time.sleep(5)
                count += 1
                continue
            
            print(f"Success: query {image_path}")
            success = True
            break
        except Exception as e:
            print(f"Error: {e}")
            count += 1
            time.sleep(5)
    
    if success:
        with open(save_json_path, "w") as f:
            json.dump(json_response, f, indent=4)
        print("Response saved to file:", save_json_path)
        
        # Extract message properly
        try:
            message = json_response['choices'][0]['message']['content']
            print("Extracted message:", message)

            # Save message to text file
            with open(save_txt_path, "w") as txt_file:
                txt_file.write(message)
            print(f"Message saved to: {save_txt_path}")

            return message
        except (KeyError, IndexError):
            print("Error extracting message from response.")
            return None

    return None

def multi_try_ensure_response_qwen(image_path, api_key, prompt, save_folder, max_tries=5, Force=False, max_new_tokens=300, max_image_size=800):
    """
    Qwen3-VL version of multi_try_ensure_response.
    Uses local Qwen3-VL model instead of OpenAI API.
    
    Args:
        image_path: Path to the image file
        api_key: Not used (kept for compatibility)
        prompt: Text prompt/question about the image
        save_folder: Folder to save response files
        max_tries: Maximum number of retry attempts (default: 5)
        Force: If True, ignore cached responses (default: False)
        max_new_tokens: Maximum tokens to generate (default: 300)
        max_image_size: Maximum image dimension (default: 1344)
    
    Returns:
        str: Generated response text, or None if failed
    """
    if not QWEN_AVAILABLE:
        raise RuntimeError("Qwen3-VL not available. Please ensure qwan.py is in the same directory.")
    
    save_json_path = os.path.join(save_folder, "response.json")
    save_txt_path = os.path.join(save_folder, "response.txt")

    if not Force:
        # Check if response.json already exists and is not empty
        if os.path.exists(save_json_path) and os.path.getsize(save_json_path) > 0:
            with open(save_json_path, "r") as f:
                try:
                    response = json.load(f)
                    # Handle both old OpenAI format and new Qwen format
                    if 'choices' in response and len(response['choices']) > 0:
                        message = response['choices'][0]['message']['content']
                    elif 'message' in response:
                        message = response['message']
                    elif 'content' in response:
                        message = response['content']
                    else:
                        message = response.get('text', str(response))
                    
                    print("output response found:", message)

                    # Save message to text file
                    with open(save_txt_path, "w") as txt_file:
                        txt_file.write(message)
                    print(f"Message saved to: {save_txt_path}")

                    return message
                except (json.JSONDecodeError, KeyError, IndexError) as e:
                    print(f"Invalid or corrupted JSON file, proceeding with new request... ({e})")

    # Try inference with retries
    count = 0
    success = False
    message = None
    
    while count < max_tries:
        try:
            print(f"Qwen3-VL inference attempt {count + 1}/{max_tries} for {image_path}")
            # Clear CUDA cache before each attempt
            import torch
            torch.cuda.empty_cache()
            
            # Use smaller image size for large images to avoid OOM
            # Check image size first
            try:
                from PIL import Image
                img = Image.open(image_path)
                img_size = max(img.size)
                img.close()
                # Use smaller max_image_size for very large images
                actual_max_size = max_image_size
                if img_size > 2000:
                    actual_max_size = 600
                elif img_size > 1500:
                    actual_max_size = 700
            except:
                actual_max_size = max_image_size
            
            message = qwen_inference(image_path, prompt, max_new_tokens=max_new_tokens, max_image_size=actual_max_size)
            
            # Clean the message - remove extra whitespace, newlines, and extract first valid category
            if message:
                message = message.strip()
                # If message contains newlines, take the first non-empty line
                lines = [line.strip() for line in message.split('\n') if line.strip()]
                if lines:
                    message = lines[0]
                # Remove quotes if present
                message = message.strip('"').strip("'").strip()
            
            if message and len(message.strip()) > 0:
                print(f"Success: query {image_path}")
                success = True
                break
            else:
                print("Empty response received, retrying...")
                count += 1
                time.sleep(2)
                
        except Exception as e:
            error_msg = str(e)
            print(f"Error during Qwen3-VL inference: {error_msg}")
            # Clear cache on error
            import torch
            torch.cuda.empty_cache()
            count += 1
            if count < max_tries:
                # Use smaller image size on retry
                max_image_size = max(400, max_image_size - 200)
                time.sleep(2)
    
    if success and message:
        # Save response in compatible format
        json_response = {
            'choices': [{
                'message': {
                    'content': message
                }
            }],
            'model': 'qwen3-vl-8b-instruct',
            'usage': {
                'prompt_tokens': 0,
                'completion_tokens': len(message.split()),
                'total_tokens': len(message.split())
            }
        }
        
        with open(save_json_path, "w") as f:
            json.dump(json_response, f, indent=4)
        print("Response saved to file:", save_json_path)
        
        # Save message to text file
        with open(save_txt_path, "w") as txt_file:
            txt_file.write(message)
        print(f"Message saved to: {save_txt_path}")

        return message
    
    print(f"Failed to get response after {max_tries} attempts")
    return None


def clean_category_response(response):
    """Clean and extract category from LLM response."""
    if not response:
        return None
    clean_message = response.replace('"', '').replace("'", '').strip()
    # If message contains newlines, take the first non-empty line
    lines = [line.strip() for line in clean_message.split('\n') if line.strip()]
    if lines:
        clean_message = lines[0]
    return clean_message


def extract_category_from_numbered_response(response):
    """Extract category name from 'number. Category_Name' format."""
    if not response:
        return None
    
    # Pattern: "1. Category_Name" or "1 Category_Name" or just "Category_Name"
    # Try to extract the category name after the number
    match = re.search(r'\d+\.?\s*(.+)', response)
    if match:
        extracted = match.group(1).strip()
        # Remove any trailing punctuation or extra text
        extracted = re.sub(r'[.:;,\s]+$', '', extracted)
        return extracted
    
    # If no number pattern, try to extract category name directly
    # Look for common category name patterns (updated to match JSON naming)
    category_patterns = [
        r'Barstool', r'Dining_Chair', r'Folding_chair', r'Lounge_Chair',
        r'Lazy_Sofa', r'L-shaped_Sofa', r'Three-Seat_Multi-seat_Sofa', r'Two-seat_Sofa',
        r'Coffee_Tea_Tables', r'Desks_Workstations', r'Meeting_Conference_Tables',
        r'Double_Bed', r'King-size_Bed', r'Single_Bed',
        r'Drawer_Chest_Corner_cabinet', r'Nightstand', r'Shelf', r'Sideboard', r'TV_Stand', r'Wardrobe'
    ]
    
    for pattern in category_patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(0)
    
    # Fallback: return cleaned response
    return clean_category_response(response)


def two_stage_category_retrieval(stitched_image_path, semantic, json_data, save_folder, max_tries=5, Force=False):
    """
    Two-stage category retrieval with description-based refinement.
    
    Stage 1: Identify parent category (dynamically from JSON)
    Stage 2: Describe the object, then match to specific subcategory
    
    Args:
        stitched_image_path: Path to stitched image with 4 viewpoints
        semantic: Semantic label (e.g., "Chair", "Table")
        json_data: JSON data containing category structure
        save_folder: Folder to save responses
        max_tries: Maximum retry attempts per stage
        Force: If True, ignore cached responses
    
    Returns:
        tuple: (final_category, stage1_response, stage2_description, stage2_response, retrieval_info)
               Returns (None, ...) if failed
    """
    # Get all parent categories from JSON dynamically
    if "OSS_cleanup" not in json_data:
        print(f"  ⚠️ Invalid JSON structure: missing 'OSS_cleanup'")
        return None, None, None, None, {}
    
    parent_categories = sorted(list(json_data["OSS_cleanup"].keys()))
    
    retrieval_info = {
        "stage1_prompt": None,
        "stage1_response": None,
        "stage2_description_prompt": None,
        "stage2_description": None,
        "stage2_match_prompt": None,
        "stage2_response": None,
        "final_category": None
    }
    
    # Build Stage 1 prompt with all parent categories
    category_list = "\n".join([f"- {cat}" for cat in parent_categories])
    stage1_prompt = prompt_stage1_parent_category_template.format(OBJECT_CATEGORY=semantic, category_list=category_list)
    retrieval_info["stage1_prompt"] = stage1_prompt
    
    stage1_save_folder = os.path.join(save_folder, "stage1_parent")
    os.makedirs(stage1_save_folder, exist_ok=True)
    
    stage1_response = multi_try_ensure_response_qwen(
        stitched_image_path, None, stage1_prompt, 
        stage1_save_folder, max_tries=max_tries, Force=Force
    )
    
    if not stage1_response:
        print(f"  ⚠️ Stage 1 failed: Could not get parent category")
        return None, None, None, None, retrieval_info
    
    parent_category = clean_category_response(stage1_response)
    retrieval_info["stage1_response"] = stage1_response
    
    # Validate parent category
    if parent_category not in parent_categories:
        print(f"  ⚠️ Stage 1 returned invalid category: {parent_category}")
        # Try to extract valid category from response
        for cat in parent_categories:
            if cat.lower() in parent_category.lower():
                parent_category = cat
                break
        else:
            return None, stage1_response, None, None, retrieval_info
    
    print(f"  ✓ Stage 1: Parent category = {parent_category}")
    
    # Check if parent category has subcategories
    has_subcategories = False
    if parent_category in json_data.get("OSS_cleanup", {}) and json_data["OSS_cleanup"][parent_category]:
        has_subcategories = True
    
    # If no subcategories or parent is "Others", return parent category
    if not has_subcategories or parent_category == "Others":
        retrieval_info["final_category"] = parent_category
        return parent_category, stage1_response, None, None, retrieval_info
    
    # Stage 2: Describe + Match to subcategory
    stage2_desc_save_folder = os.path.join(save_folder, "stage2_description")
    os.makedirs(stage2_desc_save_folder, exist_ok=True)
    
    # Step 2a: Get description
    stage2_desc_prompt = prompt_stage2_description.format(semantic, parent_category)
    retrieval_info["stage2_description_prompt"] = stage2_desc_prompt
    
    description = multi_try_ensure_response_qwen(
        stitched_image_path, None, stage2_desc_prompt,
        stage2_desc_save_folder, max_tries=max_tries, Force=True, max_new_tokens=500
    )
    
    if not description:
        print(f"  ⚠️ Stage 2a failed: Could not get description")
        # Fallback: use parent category
        retrieval_info["final_category"] = parent_category
        return parent_category, stage1_response, None, None, retrieval_info
    
    description = description.strip()
    retrieval_info["stage2_description"] = description
    print(f"  ✓ Stage 2a: Got description ({len(description)} chars)")
    
    # Step 2b: Match to subcategory using focused queries
    stage2_match_save_folder = os.path.join(save_folder, "stage2_match")
    os.makedirs(stage2_match_save_folder, exist_ok=True)
    
    # Get subcategory list - use JSON names as source of truth, match descriptions from gpt_query_summary.py
    subcategory_dict = json_data["OSS_cleanup"][parent_category]
    json_subcategory_names = list(subcategory_dict.keys())
    
    # Helper function to normalize category names for matching
    def normalize_cat_name(name):
        return name.lower().replace("_", "").replace("-", "").replace(" ", "")
    
    if SUBQUERY_AVAILABLE:
        try:
            # Get descriptions from gpt_query_summary.py
            subcategory_descriptions = retrival_query(parent_category)
            
            # Extract descriptions and map them to JSON names
            lines = subcategory_descriptions.strip().split('\n')
            description_map = {}
            for line in lines:
                # Extract number and category name from lines like "1. Category_Name: description"
                match = re.search(r'\d+\.\s*([^:]+):\s*(.+)', line)
                if match:
                    desc_cat_name = match.group(1).strip()
                    desc_text = match.group(2).strip()
                    description_map[normalize_cat_name(desc_cat_name)] = desc_text
            
            # Build subcategory list using JSON names (source of truth) with matched descriptions
            subcategory_list_parts = []
            for i, json_cat_name in enumerate(json_subcategory_names, 1):
                # Try to find matching description by normalizing both names
                normalized_json = normalize_cat_name(json_cat_name)
                desc_text = description_map.get(normalized_json)
                
                if desc_text:
                    subcategory_list_parts.append(f"\t{i}.\t{json_cat_name}: {desc_text}")
                else:
                    # No matching description found, use JSON name only
                    subcategory_list_parts.append(f"\t{i}.\t{json_cat_name}")
            
            subcategory_list = "\n".join(subcategory_list_parts)
            stage2_match_prompt = prompt_stage2_subcategory_match.format(
                parent_category, description, parent_category, subcategory_list
            )
        except Exception as e:
            print(f"  ⚠️ Could not load subcategory query: {e}")
            # Fallback: use JSON structure only
            subcategory_list = "\n".join([f"\t{i+1}.\t{cat}" for i, cat in enumerate(json_subcategory_names)])
            stage2_match_prompt = prompt_stage2_subcategory_match.format(
                parent_category, description, parent_category, subcategory_list
            )
    else:
        # Use JSON structure only
        subcategory_list = "\n".join([f"\t{i+1}.\t{cat}" for i, cat in enumerate(json_subcategory_names)])
        stage2_match_prompt = prompt_stage2_subcategory_match.format(
            parent_category, description, parent_category, subcategory_list
        )
    
    retrieval_info["stage2_match_prompt"] = stage2_match_prompt
    
    stage2_response = multi_try_ensure_response_qwen(
        stitched_image_path, None, stage2_match_prompt,
        stage2_match_save_folder, max_tries=max_tries, Force=True
    )
    
    if not stage2_response:
        print(f"  ⚠️ Stage 2b failed: Could not match subcategory")
        # Fallback: use parent category
        retrieval_info["final_category"] = parent_category
        return parent_category, stage1_response, description, None, retrieval_info
    
    retrieval_info["stage2_response"] = stage2_response
    
    # Extract final category from response
    final_category = extract_category_from_numbered_response(stage2_response)
    
    if not final_category:
        print(f"  ⚠️ Could not extract category from Stage 2 response")
        retrieval_info["final_category"] = parent_category
        return parent_category, stage1_response, description, stage2_response, retrieval_info
    
    # Validate final category exists in database
    if parent_category in json_data.get("OSS_cleanup", {}):
        valid_subcategories = list(json_data["OSS_cleanup"][parent_category].keys())
        # Try to match final_category to a valid subcategory
        matched = False
        
        # Normalize strings for comparison (lowercase, remove special chars)
        def normalize_for_match(s):
            return s.lower().replace("_", "").replace("-", "").replace(" ", "")
        
        normalized_final = normalize_for_match(final_category)
        
        # First try: exact match (case-insensitive, ignoring underscores/hyphens)
        for valid_cat in valid_subcategories:
            if normalize_for_match(valid_cat) == normalized_final:
                final_category = valid_cat
                matched = True
                break
        
        # Second try: check if the extracted category name appears in any valid category
        if not matched:
            for valid_cat in valid_subcategories:
                # Check if key parts match (e.g., "Two-seat" matches "Two-seat_Sofa")
                final_parts = [p.strip() for p in final_category.replace("_", " ").replace("-", " ").split()]
                valid_parts = [p.strip() for p in valid_cat.replace("_", " ").replace("-", " ").split()]
                
                # If all parts of final_category appear in valid_cat, it's a match
                if all(any(part.lower() in vp.lower() for vp in valid_parts) for part in final_parts):
                    final_category = valid_cat
                    matched = True
                    break
        
        # Third try: partial matching (substring match)
        if not matched:
            for valid_cat in valid_subcategories:
                normalized_valid = normalize_for_match(valid_cat)
                # Check if significant portion matches (at least 70% of shorter string)
                if len(normalized_final) > 0 and len(normalized_valid) > 0:
                    shorter = min(len(normalized_final), len(normalized_valid))
                    longer = max(len(normalized_final), len(normalized_valid))
                    if shorter / longer >= 0.7:
                        # Check if there's substantial overlap
                        overlap = sum(1 for c in normalized_final if c in normalized_valid)
                        if overlap / longer >= 0.7:
                            final_category = valid_cat
                            matched = True
                            break
        
        if not matched:
            print(f"  ⚠️ Extracted category '{final_category}' not found in database. Valid options: {valid_subcategories}. Using parent category.")
            final_category = parent_category
    
    retrieval_info["final_category"] = final_category
    print(f"  ✓ Stage 2b: Final category = {final_category}")
    
    return final_category, stage1_response, description, stage2_response, retrieval_info


def multi_try_ensure_response_qwen_multi(image_paths, api_key, prompt, save_folder, max_tries=5, Force=False, max_new_tokens=300, max_image_size=800):
    """
    Multi-image version of multi_try_ensure_response_qwen.
    Accepts multiple image paths instead of a single combined image.
    Uses local Qwen3-VL model instead of OpenAI API.
    
    Args:
        image_paths: List of paths to image files (first is stitched/context, rest are candidates)
        api_key: Not used (kept for compatibility)
        prompt: Text prompt/question about the images
        save_folder: Folder to save response files
        max_tries: Maximum number of retry attempts (default: 5)
        Force: If True, ignore cached responses (default: False)
        max_new_tokens: Maximum tokens to generate (default: 300)
        max_image_size: Maximum image dimension (default: 800)
    
    Returns:
        str: Generated response text, or None if failed
    """
    if not QWEN_AVAILABLE:
        raise RuntimeError("Qwen3-VL not available. Please ensure qwan.py is in the same directory.")
    
    if not isinstance(image_paths, list) or len(image_paths) == 0:
        raise ValueError("image_paths must be a non-empty list")
    
    # Create cache key based on image paths (use hash of sorted paths)
    import hashlib
    image_paths_str = "|".join(sorted(image_paths))
    cache_hash = hashlib.md5(image_paths_str.encode()).hexdigest()[:8]
    
    save_json_path = os.path.join(save_folder, f"response_multi_{cache_hash}.json")
    save_txt_path = os.path.join(save_folder, f"response_multi_{cache_hash}.txt")

    if not Force:
        # Check if response.json already exists and is not empty
        if os.path.exists(save_json_path) and os.path.getsize(save_json_path) > 0:
            with open(save_json_path, "r") as f:
                try:
                    response = json.load(f)
                    # Handle both old OpenAI format and new Qwen format
                    if 'choices' in response and len(response['choices']) > 0:
                        message = response['choices'][0]['message']['content']
                    elif 'message' in response:
                        message = response['message']
                    elif 'content' in response:
                        message = response['content']
                    else:
                        message = response.get('text', str(response))
                    
                    print("output response found:", message)

                    # Save message to text file
                    with open(save_txt_path, "w") as txt_file:
                        txt_file.write(message)
                    print(f"Message saved to: {save_txt_path}")

                    return message
                except (json.JSONDecodeError, KeyError, IndexError) as e:
                    print(f"Invalid or corrupted JSON file, proceeding with new request... ({e})")

    # Try inference with retries
    count = 0
    success = False
    message = None
    
    while count < max_tries:
        try:
            print(f"Qwen3-VL multi-image inference attempt {count + 1}/{max_tries} for {len(image_paths)} images")
            # Clear CUDA cache before each attempt
            import torch
            torch.cuda.empty_cache()
            
            # Use smaller image size for large images to avoid OOM
            # Check image sizes first
            try:
                from PIL import Image
                max_img_size = 0
                for img_path in image_paths:
                    img = Image.open(img_path)
                    img_size = max(img.size)
                    img.close()
                    max_img_size = max(max_img_size, img_size)
                
                # Use smaller max_image_size for very large images
                actual_max_size = max_image_size
                if max_img_size > 2000:
                    actual_max_size = 600
                elif max_img_size > 1500:
                    actual_max_size = 700
            except:
                actual_max_size = max_image_size
            
            message = qwen_inference_multi(image_paths, prompt, max_new_tokens=max_new_tokens, max_image_size=actual_max_size)
            
            # Clean the message - remove extra whitespace, newlines, and extract first valid category
            if message:
                message = message.strip()
                # If message contains newlines, take the first non-empty line
                lines = [line.strip() for line in message.split('\n') if line.strip()]
                if lines:
                    message = lines[0]
                # Remove quotes if present
                message = message.strip('"').strip("'").strip()
            
            if message and len(message.strip()) > 0:
                print(f"Success: query with {len(image_paths)} images")
                success = True
                break
            else:
                print("Empty response received, retrying...")
                count += 1
                time.sleep(2)
                
        except Exception as e:
            error_msg = str(e)
            print(f"Error during Qwen3-VL multi-image inference: {error_msg}")
            # Clear cache on error
            import torch
            torch.cuda.empty_cache()
            count += 1
            if count < max_tries:
                # Use smaller image size on retry
                max_image_size = max(400, max_image_size - 200)
                time.sleep(2)
    
    if success and message:
        # Save response in compatible format
        json_response = {
            'choices': [{
                'message': {
                    'content': message
                }
            }],
            'model': 'qwen3-vl-8b-instruct',
            'usage': {
                'prompt_tokens': 0,
                'completion_tokens': len(message.split()),
                'total_tokens': len(message.split())
            },
            'image_paths': image_paths  # Store which images were used
        }
        
        with open(save_json_path, "w") as f:
            json.dump(json_response, f, indent=4)
        print("Response saved to file:", save_json_path)
        
        # Save message to text file
        with open(save_txt_path, "w") as txt_file:
            txt_file.write(message)
        print(f"Message saved to: {save_txt_path}")

        return message
    
    print(f"Failed to get response after {max_tries} attempts")
    return None


def multi_try_ensure_response_qwen_multi_combined(image_paths, api_key, prompt, save_folder, num_votes=3, max_tries=5, Force=False, max_new_tokens=300, max_image_size=800):
    """
    Description-based strategy: Reference Description + Candidate Descriptions + Comparison with Voting
    
    Stage 1: Describe reference object (1 call) - extract geometric features
    Stage 2a: Describe each candidate independently (num_candidates calls)
    Stage 2b: Compare all candidates to reference description with voting (num_votes calls, majority vote)
    
    Args:
        image_paths: List of paths to image files [stitched_image, candidate1, candidate2, ...] (at least 2 images)
        api_key: Not used (kept for compatibility)
        prompt: Not used directly (uses internal prompts for each stage)
        save_folder: Folder to save response files
        num_votes: Number of votes for selection stage (default: 3)
        max_tries: Maximum number of retry attempts per inference (default: 5)
        Force: If True, ignore cached responses (default: False)
        max_new_tokens: Maximum tokens to generate (default: 300)
        max_image_size: Maximum image dimension (default: 800)
    
    Returns:
        tuple: (selected_template_number, detailed_stage_info_dict)
               - selected_template_number: Selected template number (1, 2, 3, or 4), or None if failed
               - detailed_stage_info_dict contains:
                 - stage1_description: The reference description text
                 - stage1_prompt: The prompt used for Stage 1
                 - stage2a_candidate_descriptions: Dict mapping candidate number to description
                 - stage2a_prompts: Dict mapping candidate number to description prompt
                 - stage2b_votes: List of votes from each round
                 - stage2b_responses: List of raw responses from each round
                 - stage2b_prompt: The comparison prompt used
                 - final_candidate: The selected candidate from voting
    """
    if not QWEN_AVAILABLE:
        raise RuntimeError("Qwen3-VL not available. Please ensure qwan.py is in the same directory.")
    
    if not isinstance(image_paths, list) or len(image_paths) < 2:
        raise ValueError("image_paths must be a list with at least 2 images (stitched + at least 1 candidate)")
    
    # Determine number of candidates
    num_candidates = len(image_paths) - 1  # Subtract 1 for stitched image
    candidate_numbers = list(range(1, num_candidates + 1))  # [1, 2, 3, ...] up to num_candidates
    
    from collections import Counter
    import re
    import torch
    
    def extract_choice(response, valid_numbers=None):
        """Extract valid choice from response."""
        if valid_numbers is None:
            valid_numbers = candidate_numbers
        
        if not response:
            return None
        response = response.strip()
        # Find all numbers in the response
        numbers = re.findall(r'\b(\d+)\b', response)
        for num_str in numbers:
            num = int(num_str)
            if num in valid_numbers:
                return num
        
        # Try patterns
        patterns = [
            r'template\s*(\d+)', r'option\s*(\d+)', r'candidate\s*(\d+)',
            r'choice\s*(\d+)', r'select\s*(\d+)', r'eliminate\s*(\d+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                num = int(match.group(1))
                if num in valid_numbers:
                    return num
        return None
    
    def safe_inference(prompt_text, stage_name, images_to_use=None, max_tokens=300):
        """Safe inference with retry logic."""
        if images_to_use is None:
            images_to_use = image_paths
        
        count = 0
        while count < max_tries:
            try:
                torch.cuda.empty_cache()
                
                # Adjust image size for large images
                try:
                    from PIL import Image
                    max_img_size = 0
                    for img_path in images_to_use:
                        img = Image.open(img_path)
                        img_size = max(img.size)
                        img.close()
                        max_img_size = max(max_img_size, img_size)
                    
                    actual_max_size = max_image_size
                    if max_img_size > 2000:
                        actual_max_size = 600
                    elif max_img_size > 1500:
                        actual_max_size = 700
                except:
                    actual_max_size = max_image_size
                
                response = qwen_inference_multi(images_to_use, prompt_text, max_new_tokens=max_tokens, max_image_size=actual_max_size)
                
                if response and len(response.strip()) > 0:
                    return response.strip()
                else:
                    count += 1
                    time.sleep(1)
            except Exception as e:
                print(f"    {stage_name} attempt {count + 1} failed: {e}")
                torch.cuda.empty_cache()
                count += 1
                if count < max_tries:
                    time.sleep(2)
        return None
    
    # ============================================
    # STAGE 1: REFERENCE DESCRIPTION
    # ============================================
    print(f"  Stage 1: Describing reference object...")
    
    # Use only the stitched image (first image) for description
    reference_image = [image_paths[0]]
    reference_description = safe_inference(Prompt_reference_description, "Reference description", images_to_use=reference_image, max_tokens=200)
    
    if not reference_description:
        print(f"  → Failed to get reference description, falling back to direct comparison")
        reference_description = "Focus on geometric structure, shape, and form."
    
    print(f"  → Reference description obtained ({len(reference_description)} chars)")
    
    # Store Stage 1 info
    stage_info = {
        "stage1_description": reference_description,
        "stage1_prompt": Prompt_reference_description,
    }
    
    # ============================================
    # STAGE 2a: DESCRIBE EACH CANDIDATE INDEPENDENTLY
    # ============================================
    print(f"  Stage 2a: Describing each candidate independently...")
    
    candidate_descriptions = {}
    stage2a_prompts = {}
    
    for candidate_num in candidate_numbers:
        candidate_idx = candidate_num - 1  # Convert to 0-based index (candidate 1 -> index 0)
        candidate_image = [image_paths[candidate_idx + 1]]  # +1 because image_paths[0] is reference
        
        candidate_desc_prompt = f"""
You are given Image {candidate_num + 1}: A 3D template candidate labeled {candidate_num}.

TASK: Describe the GEOMETRIC STRUCTURE of this template. Completely ignore color, texture, material, and surface appearance.

Describe the following geometric aspects:
1. STRUCTURAL ELEMENTS: What are the main components/parts of this template? (e.g., base, top, supports, arms, back, legs, etc.)
2. OVERALL SHAPE: What is the general form? (rectangular, circular, curved, angular, organic, etc.)
3. PROPORTIONS: How do the components relate to each other in size and position? (e.g., wide base, tall supports, etc.)
4. GEOMETRIC DETAILS: Any distinctive features? (curves, angles, edges, symmetry, asymmetry, etc.)
5. PROFILE/SILHOUETTE: What does the overall outline look like from different viewpoints?

Be specific and focus ONLY on geometry and structure. Do not mention color, material, or texture.
"""
        
        desc = safe_inference(candidate_desc_prompt, f"Candidate {candidate_num} description", 
                              images_to_use=candidate_image, max_tokens=200)
        if desc:
            candidate_descriptions[candidate_num] = desc.strip()
            stage2a_prompts[candidate_num] = candidate_desc_prompt
            print(f"    → Candidate {candidate_num} description obtained ({len(desc)} chars)")
        else:
            candidate_descriptions[candidate_num] = f"Template {candidate_num} geometric structure"
            stage2a_prompts[candidate_num] = candidate_desc_prompt
            print(f"    → Candidate {candidate_num} description failed, using placeholder")
    
    # Store Stage 2a info
    stage_info["stage2a_candidate_descriptions"] = candidate_descriptions
    stage_info["stage2a_prompts"] = stage2a_prompts
    
    # ============================================
    # STAGE 2b: COMPARISON & SELECTION (WITH VOTING)
    # ============================================
    print(f"  Stage 2b: Comparing candidates with voting (running {num_votes} times)...")
    
    # Create comparison prompt with reference description and candidate descriptions
    candidate_list_str = ", ".join(map(str, candidate_numbers))
    candidate_descriptions_text = "\n".join([f"Template {num}:\n{desc}\n" for num, desc in candidate_descriptions.items()])
    
    comparison_prompt = f"""
You are given {len(image_paths)} images:
- Image 1: A stitched image showing four real RGB images of an object from different viewpoints (the reference object)
- Images 2-{len(image_paths)}: {num_candidates} 3D template candidate(s) labeled {candidate_list_str} respectively

REFERENCE OBJECT DESCRIPTION:
{reference_description}

CANDIDATE DESCRIPTIONS:
{candidate_descriptions_text}

TASK: Compare each template ({candidate_list_str}) to the reference description above, considering both the images and their descriptions.

For each template, evaluate:
- Does it match the structural elements described in the reference?
- Does it match the overall shape and form?
- Does it match the proportions and component relationships?
- Does it match the geometric details (curves, angles, edges)?

IMPORTANT: Ignore color, texture, and material completely. Focus ONLY on geometry. Objects may be scaled, so absolute dimensions are not important.

Select the template ({candidate_list_str}) that best matches the reference description.

Output ONLY the number ({candidate_list_str}).
"""
    
    selection_votes = []
    stage2b_responses = []  # Store raw responses
    for vote_round in range(num_votes):
        response = safe_inference(comparison_prompt, f"Comparison round {vote_round + 1}", max_tokens=max_new_tokens)
        if response:
            stage2b_responses.append(response)  # Save raw response
            choice = extract_choice(response, candidate_numbers)
            if choice and choice in candidate_numbers:
                selection_votes.append(choice)
                print(f"    Comparison round {vote_round + 1}: Template {choice}")
            else:
                print(f"    Comparison round {vote_round + 1}: Invalid response '{response}', extracted: {choice}")
        else:
            stage2b_responses.append(None)  # Track failed rounds
    
    # Store Stage 2b info
    stage_info["stage2b_votes"] = selection_votes
    stage_info["stage2b_responses"] = stage2b_responses
    stage_info["stage2b_prompt"] = comparison_prompt
    
    if not selection_votes:
        # Fallback: use first candidate
        final_candidate = candidate_numbers[0]
        print(f"  → No valid selections, using default: Template {final_candidate}")
    else:
        # Majority vote
        vote_counts = Counter(selection_votes)
        final_candidate = vote_counts.most_common(1)[0][0]
        print(f"  → Majority selection: Template {final_candidate} ({vote_counts[final_candidate]}/{len(selection_votes)} votes)")
    
    stage_info["final_candidate"] = final_candidate
    
    # Return final candidate (no verification stage)
    print(f"  → Final selection: Template {final_candidate}")
    return str(final_candidate), stage_info


def find_subfolder(base_folder, subfolder_name):
    """
    Searches 'base_folder' (recursively, following symbolic links) for a subfolder called 'subfolder_name'.
    Returns the full path to the first matching subfolder if found, otherwise None.
    """
    for root, dirs, files in os.walk(base_folder, followlinks=True):  # Follow symbolic links
        if dirs == []:  # Skip directories without subfolders
            continue
        if subfolder_name in dirs:  # Check if subfolder_name is in the list of directories
            return os.path.join(root, subfolder_name)  # Return full path to the subfolder
    return None  # Return None if not found
