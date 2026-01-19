"""
Material Selection Prompts for Material Painting Pipeline

This module contains prompts related to material type selection,
refinement, and matching for the material painting pipeline.
"""

from typing import List, Optional


# =============================================================================
# Valid Materials Constants
# =============================================================================

VALID_MATERIALS = [
    'Blends', 'Ceramic', 'Concrete', 'Fabric', 'Ground',
    'Leather', 'Marble', 'Metal', 'Plaster', 'Plastic',
    'Stone', 'Terracotta', 'Wood', 'Misc'
]
"""List of valid material categories for the pipeline."""

MATERIAL_MAPPING = {
    "Rubber": "Plastic",
    "Glass": "Metal",
    "Bone": "Marble"
}
"""Mapping of alternate material names to valid categories."""


# =============================================================================
# Material Refinement Prompts
# =============================================================================

PROMPT_REFINE = """{}\n\nSelect the most similar {} material type of number {} part of the image, according to the analysis of corresponding part material(including color, pattern, roughness, age and so on...). If you find it difficult to subdivide, just output {}. Don't output other information. Only a single word representing the category from optional list needs to be output. (optional list of material is {}). """
"""
Material type selection prompt for narrowing down material subcategories.

Args (format):
    intro_text (str): Introduction/context text
    material_type (str): Main material type
    part_number (str): Part identifier number
    fallback (str): Fallback material type
    options_list (str): Available material options
"""

PROMPT_REFINE_NEW = """{} Select the most similar {} material type of this red segmentation of the 3d mesh, according to the analysis of corresponding part material(including color, pattern, roughness, age and so on...). Don't output other information. Only a single word representing the category from optional list needs to be output. (optional list of material is {}). """
"""
Enhanced material type selection prompt for red segmentation analysis.

Args (format):
    intro_text (str): Introduction/context text
    material_type (str): Main material type
    options_list (str): Available material options
"""

PROMPT_REFINE_SUB = """{}\n\nLook at the material carefully of number {} part of the image, here are some descriptions about {} materials, can you tell me which is the best description match the part {} in the image?\n{}
Just tell me the final result in dict format with material name and descrption. Don't output other information.
"""
"""
Material description matching prompt for detailed material selection.

Args (format):
    intro_text (str): Introduction/context text
    part_number (str): Part identifier number
    material_type (str): Main material type
    part_id (str): Part identifier
    descriptions (str): Formatted material descriptions
"""

PROMPT_REFINE_SUB_OVERALL = """{}

Carefully examine the material in the red-segmented area for its potential correspondence in the  top row captured image. Here are some descriptions of {} materials. Can you identify the best description that matches the material in the this part of the original capture?

{}

Please provide only the final result in dictionary format with the material name and description, not a json file format. Do not include any other information.
"""
"""
Advanced material description matching prompt for overall part analysis.

Args (format):
    intro_text (str): Introduction/context text
    material_type (str): Main material type
    descriptions (str): Formatted material descriptions
"""


# =============================================================================
# Helper Functions
# =============================================================================

def check_material_valid(material: str, retry_count: int) -> Optional[str]:
    """
    Validate and normalize a material name.

    Args:
        material: Material name to validate
        retry_count: Current retry count (fallback to 'Misc' after 3)

    Returns:
        Valid material name, mapped material name, 'Misc' (after 3 retries), or None
    """
    if material in VALID_MATERIALS:
        return material
    if material in MATERIAL_MAPPING:
        return MATERIAL_MAPPING[material]
    if retry_count > 3:
        return 'Misc'
    return None


def get_refine_prompt(
    intro_text: str,
    material_type: str,
    part_number: str,
    fallback: str,
    options_list: List[str]
) -> str:
    """
    Get the material refinement prompt.

    Args:
        intro_text: Introduction/context text
        material_type: Main material type
        part_number: Part identifier number
        fallback: Fallback material type
        options_list: List of available material options

    Returns:
        Formatted refinement prompt string
    """
    options_str = ", ".join(options_list) if isinstance(options_list, list) else str(options_list)
    return PROMPT_REFINE.format(intro_text, material_type, part_number, fallback, options_str)


def get_refine_new_prompt(
    intro_text: str,
    material_type: str,
    options_list: List[str]
) -> str:
    """
    Get the enhanced material refinement prompt.

    Args:
        intro_text: Introduction/context text
        material_type: Main material type
        options_list: List of available material options

    Returns:
        Formatted refinement prompt string
    """
    options_str = ", ".join(options_list) if isinstance(options_list, list) else str(options_list)
    return PROMPT_REFINE_NEW.format(intro_text, material_type, options_str)


def get_refine_sub_prompt(
    intro_text: str,
    part_number: str,
    material_type: str,
    part_id: str,
    descriptions: str
) -> str:
    """
    Get the material description matching prompt.

    Args:
        intro_text: Introduction/context text
        part_number: Part identifier number
        material_type: Main material type
        part_id: Part identifier
        descriptions: Formatted material descriptions

    Returns:
        Formatted description matching prompt string
    """
    return PROMPT_REFINE_SUB.format(intro_text, part_number, material_type, part_id, descriptions)


def get_refine_sub_overall_prompt(
    intro_text: str,
    material_type: str,
    descriptions: str
) -> str:
    """
    Get the advanced material description matching prompt.

    Args:
        intro_text: Introduction/context text
        material_type: Main material type
        descriptions: Formatted material descriptions

    Returns:
        Formatted overall matching prompt string
    """
    return PROMPT_REFINE_SUB_OVERALL.format(intro_text, material_type, descriptions)


def get_valid_materials() -> List[str]:
    """
    Get the list of valid material categories.

    Returns:
        List of valid material category names
    """
    return VALID_MATERIALS.copy()


def get_material_mapping() -> dict:
    """
    Get the material name mapping dictionary.

    Returns:
        Dictionary mapping alternate names to valid categories
    """
    return MATERIAL_MAPPING.copy()
