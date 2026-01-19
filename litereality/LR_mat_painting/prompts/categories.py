"""
Furniture Category Query Prompts for Material Painting Pipeline

This module contains prompts for identifying furniture subcategories,
used during the retrieval phase of the material painting pipeline.
"""

from typing import Optional


# =============================================================================
# Subcategory Selection Prompt
# =============================================================================

QUERY_SUB_CLASS = """
This is a set of images of a {} object, cropped from photos taken during an iPhone scan. The images may include background elements from a real room environment. For the following task, it is crucial to focus solely on the main {} object and disregard any surrounding elements. Within the {} categories, we have a list of more detailed subcategories as follows. Could you please identify the most suitable subcategory for the object in the image? Directly output the category number and its name without any additional information. The subcategories include:
{}
"""
"""
Subcategory selection prompt template.

Args (format):
    type_str (str): Object type (e.g., "Chair")
    type_str (str): Object type repeated
    type_str (str): Object type repeated
    subcategories (str): Formatted list of subcategories
"""


# =============================================================================
# Furniture-Specific Subcategory Lists
# =============================================================================

CHAIR_QUERY = """
	1.	Barstool: High-seated stools, often used at bars or high tables, usually with a footrest and sometimes backless or with a low backrest.
	2.	Dining_Chair: Chairs for dining tables, typically with balanced height and a comfortable backrest for dining.
	3.	Folding_chair: Foldable chair, lightweight and easy to store.
	4.	Lounge_Chair_Book-chair_Computer_Chair: Comfortable lounge or computer chair, designed for relaxation or extended computer use.
	5.	Lounge_Chair_Cafe_Chair_Office_Chair: Comfortable lounge or office chair, suitable for cafes, office, or casual seating.
"""
"""Chair subcategory definitions."""

SOFA_QUERY = """
	1.	Lazy_Sofa: A comfortable, casual sofa for lounging, typically featuring soft cushions and a low profile.
	2.	Three-Seat_Multi-seat_Sofa: A sofa designed for three people or more, ideal for living rooms or larger spaces, often providing a balanced mix of seating capacity and comfort.
	3.	Two-seat_Sofa: A compact sofa designed for two people, suitable for smaller rooms or cozy spaces where space is limited.
"""
"""Sofa subcategory definitions."""

TABLE_QUERY = """
	1.	Coffee_Tea_Tables: This category includes all small, low-to-medium height surface furniture. It encompasses traditional coffee tables, tea tables, and side/end tables (often placed next to sofas or beds). Crucially, include accent tables and pedestal stands specifically designed to hold items on top—such as lamps, drinks, books, or decorative objects. If a piece of furniture is designed as a small, waist-high or lower horizontal surface for supporting objects in a lounge or seating context, it belongs in this cluster.
	2.	Desks_Workstations: Tables designed for individual work or study, featuring ergonomic height (approximately 70-75 cm), often with drawers, cable management, or rectangular tops meant for one person. Used in home offices, corporate workspaces, or study nooks.
	3.	Meeting_Conference_Tables: Larger tables intended for group use, typically seating multiple people. Used in meeting rooms, conference rooms, or dining areas. Designed for collaborative work or group dining.
"""
"""Table subcategory definitions."""

BED_QUERY = """
	1.	Double_Bed: A bed designed to accommodate two people, slightly smaller than a queen, ideal for guest rooms or smaller bedrooms.
	2.	King-size_Bed: A large bed with ample space for two people, offering maximum comfort and room, typically found in master bedrooms.
	3.	Single_Bed: A compact bed intended for one person, perfect for small rooms or spaces like guest rooms and children's bedrooms.
"""
"""Bed subcategory definitions."""

STORAGE_QUERY = """
	1.	Drawer_Chest_Corner_cabinet: A storage unit with drawers, typically designed to fit in a corner, suitable for storing clothing, linens, or household items.
	2.	Nightstand: A small bedside table or cabinet, used for holding items like lamps, books, or personal belongings.
	3.	Shelf: A wall-mounted or freestanding unit with horizontal surfaces, used for displaying or organizing books, decor, or other items.
	4.	Sideboard_Side_Cabinet_Console: A low, long storage unit with cabinets and sometimes drawers, commonly used in dining rooms or entryways for storing tableware, linens, or decor.
	5.	TV_Stand: A stand designed to support a television, usually with compartments or shelves for media players, consoles, and storage.
	6.	Wardrobe: A tall, freestanding cabinet for storing clothing, featuring hanging space, shelves, and sometimes drawers.
"""
"""Storage furniture subcategory definitions."""


# =============================================================================
# Category Query Mapping
# =============================================================================

CATEGORY_QUERIES = {
    "Chair": CHAIR_QUERY,
    "Sofa": SOFA_QUERY,
    "Table": TABLE_QUERY,
    "Bed": BED_QUERY,
    "Storage": STORAGE_QUERY,
}
"""Mapping of furniture types to their subcategory query strings."""


# =============================================================================
# Helper Functions
# =============================================================================

def get_retrieval_query(type_str: str) -> str:
    """
    Get the subcategory query string for a given furniture type.

    Args:
        type_str: Furniture type (e.g., "Chair", "Sofa", "Table", "Bed", "Storage")

    Returns:
        Subcategory query string for the given type

    Raises:
        ValueError: If the object type is not recognized
    """
    if type_str not in CATEGORY_QUERIES:
        raise ValueError(
            f"Invalid object type: {type_str}. "
            f"Valid types are: {', '.join(CATEGORY_QUERIES.keys())}"
        )
    return CATEGORY_QUERIES[type_str]


def get_subcategory_prompt(type_str: str) -> str:
    """
    Get the complete subcategory selection prompt for a furniture type.

    Args:
        type_str: Furniture type (e.g., "Chair", "Sofa")

    Returns:
        Formatted subcategory selection prompt

    Raises:
        ValueError: If the object type is not recognized
    """
    subcategories = get_retrieval_query(type_str)
    return QUERY_SUB_CLASS.format(type_str, type_str, type_str, subcategories)


def get_supported_furniture_types() -> list:
    """
    Get the list of supported furniture types.

    Returns:
        List of supported furniture type names
    """
    return list(CATEGORY_QUERIES.keys())


def is_valid_furniture_type(type_str: str) -> bool:
    """
    Check if a furniture type is supported.

    Args:
        type_str: Furniture type to check

    Returns:
        True if the type is supported, False otherwise
    """
    return type_str in CATEGORY_QUERIES


# =============================================================================
# Backward Compatibility
# =============================================================================

# Legacy function name for backward compatibility
retrieval_query = get_retrieval_query
"""Legacy alias for get_retrieval_query (backward compatibility)."""

# Legacy constant names
query_sub_class = QUERY_SUB_CLASS
chair_query = CHAIR_QUERY
sofa_query = SOFA_QUERY
table_query = TABLE_QUERY
bed_query = BED_QUERY
storage_query = STORAGE_QUERY
