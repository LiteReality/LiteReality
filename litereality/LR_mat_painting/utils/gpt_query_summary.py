"""
Furniture Category Query Module for Material Painting Pipeline

This module provides functions and prompts for identifying furniture subcategories
during the retrieval phase. It serves as a compatibility wrapper for the centralized
prompts module.

Usage:
    from litereality.LR_mat_painting.utils.gpt_query_summary import retrival_query

    query = retrival_query("Chair")  # Get chair subcategory query
"""

# Import all category-related prompts from centralized module
from litereality.LR_mat_painting.prompts.categories import (
    # Prompt template
    QUERY_SUB_CLASS as query_sub_class,
    # Furniture-specific queries
    CHAIR_QUERY as chair_query,
    SOFA_QUERY as sofa_query,
    TABLE_QUERY as table_query,
    BED_QUERY as bed_query,
    STORAGE_QUERY as storage_query,
    # Functions
    get_retrieval_query,
)


def retrival_query(type_str):
    """
    Get the subcategory query string for a furniture type.

    This function is a compatibility wrapper for get_retrieval_query.

    Args:
        type_str: Furniture type (e.g., "Chair", "Sofa", "Table", "Bed", "Storage")

    Returns:
        str: Subcategory query string for the given type

    Raises:
        Exception: If the object type is not recognized
    """
    try:
        return get_retrieval_query(type_str)
    except ValueError as e:
        # Re-raise as Exception for backward compatibility
        raise Exception(str(e))
