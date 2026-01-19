"""
VLM Phase-Specific Prompts for Material Painting Pipeline

This module contains prompt generators for the various VLM (Vision-Language Model)
phases in the material selection pipeline:
- Phase 1: Holistic color analysis
- Phase 2: Pattern analysis
- Phase 3: Color adaptation (handled in Material_refinements.py)
- Phase 4: Final material selection
"""

from typing import List, Optional


# =============================================================================
# Phase 1: Holistic Color Analysis
# =============================================================================

def get_holistic_color_prompt(type_str: str, num_images: int) -> str:
    """
    Generate the holistic color analysis prompt for Phase 1.

    This prompt analyzes multiple captured images to identify main colors
    and material composition across the entire object.

    Args:
        type_str: Object type (e.g., "Chair", "Table")
        num_images: Number of images being analyzed

    Returns:
        Formatted holistic color analysis prompt
    """
    return f"""You are analyzing a {type_str or "furniture object"} to identify its main colors and material composition.

IMAGES PROVIDED:
You are given {num_images} photographs of the same {type_str or "object"} captured from different angles. These images show the {type_str or "object"} in a real room environment.

TASK:
Analyze all {num_images} images together and identify:
1. The main colors present in this {type_str or "object"}
2. Which parts/components have which colors
3. The overall color theme

INSTRUCTIONS:
1. Examine all {num_images} images to get a comprehensive view of the {type_str or "object"}
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


# =============================================================================
# Phase 2: Pattern Analysis
# =============================================================================

def get_pattern_analysis_prompt(material_ids: List[str], object_type: str = "object") -> str:
    """
    Generate the pattern analysis prompt for Phase 2.

    This prompt analyzes albedo patterns of material candidates and selects
    the top 4 best matches based on texture similarity.

    Args:
        material_ids: List of material IDs to analyze
        object_type: Object type for context

    Returns:
        Formatted pattern analysis prompt
    """
    material_list_str = ', '.join(material_ids)

    return f"""Analyze these material textures against the reference images:

Top row: [Reference photo] [3D model rendering of the part]
Bottom rows: {len(material_ids)} material texture options (albedo maps) arranged in a grid

Material IDs (in order): {material_list_str}

Task: Select the TOP 4 material textures that have the best pattern/texture match to the part shown in the reference photo.

Consider:
- Texture pattern similarity (grain direction, scale, detail level)
- Pattern appropriateness for the part geometry
- Surface detail matching (roughness, smoothness)
- Overall visual coherence

Return format (JSON):
{{
  "top_4": ["material_id_1", "material_id_2", "material_id_3", "material_id_4"],
  "scores": {{"material_id_1": 0.95, "material_id_2": 0.92, ...}},
  "reasoning": "Why these 4 materials have the best pattern matches"
}}"""


# =============================================================================
# Phase 4: Final Material Selection
# =============================================================================

def get_final_selection_prompt(material_ids: List[str], object_type: str = "object") -> str:
    """
    Generate the final selection prompt for Phase 4.

    This prompt selects the single best material from color-adapted candidates
    based on complete visual analysis.

    Args:
        material_ids: List of material IDs (up to 4 candidates)
        object_type: Object type for context

    Returns:
        Formatted final selection prompt
    """
    # Pad the list to 4 entries for consistent formatting
    padded_ids = material_ids[:4] + ['N/A'] * (4 - len(material_ids))

    return f"""Compare these color-adapted material textures against the reference images:

Top row: [Reference photo] [3D model rendering of the part]
Bottom row: 4 color-adapted material options labeled 1-4

Material IDs (in order):
1. {padded_ids[0]}
2. {padded_ids[1]}
3. {padded_ids[2]}
4. {padded_ids[3]}

Task: Select the single best material texture that most closely matches the appearance of the part in the reference photo after color adaptation.

Consider:
- Overall visual match to reference
- How well color adaptation worked
- Pattern/texture appropriateness after color change
- Realism and coherence

Return format (JSON):
{{
  "selected_material": "material_id",
  "confidence": 0-10,
  "reasoning": "Why this material best matches the reference after color adaptation"
}}"""


# =============================================================================
# Part Identification Prompts
# =============================================================================

def get_part_identification_prompt(type_str: str) -> str:
    """
    Generate the part identification prompt for Step 2.

    This prompt identifies and describes segmented parts from 3D mesh images.

    Args:
        type_str: Object type (e.g., "Chair")

    Returns:
        Formatted part identification prompt
    """
    return f"""You are analyzing a segmented part from a {type_str or "3D object"}.

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


# =============================================================================
# Color Voting Prompts
# =============================================================================

def get_color_voting_prompt(
    type_str: str,
    part_description: str,
    holistic_context: str,
    variant: int = 0
) -> str:
    """
    Generate color voting prompt for Step 3.

    This prompt extracts RGB color values through voting across multiple images.

    Args:
        type_str: Object type (e.g., "Chair")
        part_description: Description of the part being analyzed
        holistic_context: Formatted holistic context from Step 1
        variant: Prompt variant index (0-3) for voting diversity

    Returns:
        Formatted color voting prompt
    """
    base_prompt = f"""Color estimation for "{part_description}" segment of a {type_str or "furniture object"}.

HOLISTIC OBJECT CONTEXT (from Step 1):
{holistic_context}

INPUT IMAGES:
- Image 1: Real photographs of the {type_str or "object"} captured from multiple angles (stitched_image.jpg)
- Image 2: 3D renderings with this part highlighted in red{"" if variant == 0 else " from a different angle" if variant == 1 else ""}

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

    return base_prompt


# =============================================================================
# Top 10 Material Selection Prompts
# =============================================================================

def get_top_10_selection_prompt(
    intro_prompt: str,
    object_type: str,
    material_type: str,
    material_list_text: str,
    num_to_select: int
) -> str:
    """
    Generate the top 10 material selection prompt for Phase 1.

    This prompt selects the top N material candidates from database descriptions.

    Args:
        intro_prompt: Introduction prompt with context
        object_type: Object type (e.g., "Chair")
        material_type: Material subcategory type
        material_list_text: Formatted list of material IDs and descriptions
        num_to_select: Number of materials to select

    Returns:
        Formatted top 10 selection prompt
    """
    return f"""Image 1 shows the full captured {object_type} object. Image 2 shows the segmented part mask for this specific part.

{intro_prompt}

From the detailed descriptions of {material_type} materials below, select the TOP {num_to_select} most appropriate materials for the part shown in Image 2, considering how it appears in the full object context of Image 1:

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


# =============================================================================
# Semantic Mapping Prompts
# =============================================================================

def get_semantic_mapping_prompt(
    object_type: str,
    part_description: str,
    holistic_parts_str: str
) -> str:
    """
    Generate the semantic mapping prompt for color validation.

    This prompt maps segmented part descriptions to holistic parts from Step 1.

    Args:
        object_type: Object type (e.g., "Chair")
        part_description: Description of the segmented part
        holistic_parts_str: Formatted string of holistic parts and colors

    Returns:
        Formatted semantic mapping prompt
    """
    return f"""You are mapping a segmented part from a {object_type} to the holistic parts identified in Step 1.

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


# =============================================================================
# Color Verification Prompts
# =============================================================================

def get_color_verification_prompt(
    object_type: str,
    holistic_rgb: List[int],
    per_part_rgb: List[int],
    part_description: str,
    holistic_part: str,
    overall_theme: str
) -> str:
    """
    Generate the color verification prompt for Step 4 validation.

    This prompt verifies which color estimate (holistic vs per-part) is more accurate.

    Args:
        object_type: Object type (e.g., "Chair")
        holistic_rgb: RGB color from holistic analysis
        per_part_rgb: RGB color from per-part analysis
        part_description: Description of the part
        holistic_part: Name of matched holistic part
        overall_theme: Overall color theme from holistic analysis

    Returns:
        Formatted color verification prompt
    """
    return f"""You are comparing two color estimates for a {object_type} part.

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
