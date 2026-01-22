"""
Prompts Package for Material Painting Pipeline

This package contains all prompt templates and generators used throughout
the material painting pipeline, organized by functionality:
- base.py: Core prompt templates
- materials.py: Material selection and refinement prompts
- categories.py: Furniture category query prompts
- vlm_phases.py: VLM phase-specific prompt generators
"""

# Import from base.py
from .base import (
    ADD_PROMPT,
    ADD_PROMPT_DOUBLE,
    PROMPT_OVERALL,
    INTRODUCTION_PROMPT,
    SEGMENTATION_PROMPT,
    SEGMENT_GROUP_PROMPT,
    COLOR_INFORMATION_EXTRACTION,
    PROMPT_SHAPE,
    QUERY_CROPPING,
    get_object_intro_prompt,
    get_dual_section_intro_prompt,
    get_introduction_prompt,
    get_segmentation_prompt,
    get_segment_group_prompt,
    get_color_extraction_prompt,
    get_crop_selection_prompt,
)

# Import from materials.py
from .materials import (
    VALID_MATERIALS,
    MATERIAL_MAPPING,
    PROMPT_REFINE,
    PROMPT_REFINE_NEW,
    PROMPT_REFINE_SUB,
    PROMPT_REFINE_SUB_OVERALL,
    check_material_valid,
    get_refine_prompt,
    get_refine_new_prompt,
    get_refine_sub_prompt,
    get_refine_sub_overall_prompt,
    get_valid_materials,
    get_material_mapping,
)

# Import from categories.py
from .categories import (
    QUERY_SUB_CLASS,
    CHAIR_QUERY,
    SOFA_QUERY,
    TABLE_QUERY,
    BED_QUERY,
    STORAGE_QUERY,
    CATEGORY_QUERIES,
    get_retrieval_query,
    get_subcategory_prompt,
    get_supported_furniture_types,
    is_valid_furniture_type,
    # Backward compatibility aliases
    retrieval_query,
    query_sub_class,
    chair_query,
    sofa_query,
    table_query,
    bed_query,
    storage_query,
)

# Import from vlm_phases.py
from .vlm_phases import (
    get_holistic_color_prompt,
    get_pattern_analysis_prompt,
    get_final_selection_prompt,
    get_part_identification_prompt,
    get_color_voting_prompt,
    get_top_10_selection_prompt,
    get_semantic_mapping_prompt,
    get_color_verification_prompt,
)

# Export all public symbols
__all__ = [
    # Base prompts
    "ADD_PROMPT",
    "ADD_PROMPT_DOUBLE",
    "PROMPT_OVERALL",
    "INTRODUCTION_PROMPT",
    "SEGMENTATION_PROMPT",
    "SEGMENT_GROUP_PROMPT",
    "COLOR_INFORMATION_EXTRACTION",
    "PROMPT_SHAPE",
    "QUERY_CROPPING",
    # Base helper functions
    "get_object_intro_prompt",
    "get_dual_section_intro_prompt",
    "get_introduction_prompt",
    "get_segmentation_prompt",
    "get_segment_group_prompt",
    "get_color_extraction_prompt",
    "get_crop_selection_prompt",
    # Material constants
    "VALID_MATERIALS",
    "MATERIAL_MAPPING",
    "PROMPT_REFINE",
    "PROMPT_REFINE_NEW",
    "PROMPT_REFINE_SUB",
    "PROMPT_REFINE_SUB_OVERALL",
    # Material helper functions
    "check_material_valid",
    "get_refine_prompt",
    "get_refine_new_prompt",
    "get_refine_sub_prompt",
    "get_refine_sub_overall_prompt",
    "get_valid_materials",
    "get_material_mapping",
    # Category constants
    "QUERY_SUB_CLASS",
    "CHAIR_QUERY",
    "SOFA_QUERY",
    "TABLE_QUERY",
    "BED_QUERY",
    "STORAGE_QUERY",
    "CATEGORY_QUERIES",
    # Category helper functions
    "get_retrieval_query",
    "get_subcategory_prompt",
    "get_supported_furniture_types",
    "is_valid_furniture_type",
    # Backward compatibility
    "retrieval_query",
    "query_sub_class",
    "chair_query",
    "sofa_query",
    "table_query",
    "bed_query",
    "storage_query",
    # VLM phase functions
    "get_holistic_color_prompt",
    "get_pattern_analysis_prompt",
    "get_final_selection_prompt",
    "get_part_identification_prompt",
    "get_color_voting_prompt",
    "get_top_10_selection_prompt",
    "get_semantic_mapping_prompt",
    "get_color_verification_prompt",
]
