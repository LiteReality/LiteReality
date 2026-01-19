"""
Base Prompt Templates for Material Painting Pipeline

This module contains core prompt templates used for object introduction,
segmentation analysis, and shape queries in the material painting pipeline.
"""

# =============================================================================
# Object Introduction Prompts
# =============================================================================

ADD_PROMPT = (
    "\nThis is a set of images of a {} object, cropped from photos taken during "
    "an iPhone scan. The images may include background elements from a real room "
    "environment; please it is very important for the following task to focus solely "
    "on the main {} object and disregard any surrounding elements. "
)
"""
Single object introduction prompt.

Args (format):
    type_str (str): Object type (e.g., "Chair", "Table")
    type_str (str): Object type repeated for emphasis

Example:
    >>> prompt = ADD_PROMPT.format("Chair", "Chair")
"""

ADD_PROMPT_DOUBLE = (
    "\n This image consists of two sections: on the left side, a series of images "
    "show a {} object, cropped from photos taken during an iPhone scan. While some "
    "background elements from a real room may be visible, the focus should remain "
    "solely on the main {} object, disregarding any surrounding items. On the right "
    "side, there are four multiview images of a segmented 3D mesh model, where each "
    "color corresponds to a different segment, and a central legend lists the name "
    "of each segment. For this particular 3D model of {}, there are {} parts, "
    "including {}. Our goal is to assign materials to each part of the 3D mesh so "
    "that it visually aligns with the appearance of the object shown in the images "
    "on the left. "
)
"""
Dual-section introduction prompt for images with both photos and 3D mesh views.

Args (format):
    type_str (str): Object type
    type_str (str): Object type repeated
    type_str (str): Object type repeated
    num_parts (int): Number of parts in the mesh
    parts_list (str): Comma-separated list of part names
"""


# =============================================================================
# Material Identification Prompts
# =============================================================================

PROMPT_OVERALL = """ \n
To complete this task, identify the material for that part, based on the red segmented areas. Use the following format for the output: {material name} only the material name without any other words, punctuation, or explanations. Available materials are as follows: Ceramic, Concrete, Fabric, Ground, Leather, Marble, Metal, Plaster, Plastic, Stone, Terracotta, Wood, Misc (use "Misc" if no other material fits)
"""
"""
Material identification instruction prompt.
Used for identifying the main material category of a segmented part.
"""


# =============================================================================
# Image Context Prompts
# =============================================================================

INTRODUCTION_PROMPT = """\n
Image 1 displays photos captured from an indoor environment, cropped to focus on showing a {} in a real room - please disregard any background elements in the photos. Image 2 contains the segmentation mask for the interested region highlighted in red, which is described as {}. Although the segmentation may not be an exact match to the photographed object, it is very similar, allowing for effective material matching. Our goal is to assign materials to the red-highlighted part by analyzing how it appears in the context of the full object shown in Image 1."""
"""
Introduction prompt for two-image context (photo + segmentation mask).

Args (format):
    type_str (str): Object type (e.g., "Chair")
    description (str): Description of the red-highlighted segment
"""


# =============================================================================
# Segmentation Prompts
# =============================================================================

SEGMENTATION_PROMPT = """
These are four sets of images of a rendered 3D mesh model of {} object,  with red-highlighted segmentation masks. In each set, the left image shows the rendered view, while the right image shows the segmentation mask, with the red area marking the region of interest. Four different views of the same segmentation are provided. Refer to the original image to understand the red segment, and describe the component and its potential materials. Output the description directly in 20 words or less. If the red highlights are very small, observe carefully.
"""
"""
Segmentation description prompt.
Used to describe a part's component and potential materials from segmentation masks.

Args (format):
    type_str (str): Object type
"""

SEGMENT_GROUP_PROMPT = """
This image depicts indoor furniture, divided into two main sections. The upper section focuses on segmented masks, where each mask highlights a specific region in red within an indexed area. Only these red segments are of interest for this task. The following is the description of the red segmented parts:

{}

The goal is to analyze the red-highlighted segments and determine a suitable way to group them by similarity while avoiding over-grouping. For instance, some segments, such as handles, have distinct characteristics and should remain in separate groups.

The output should focus on how these red segments can be effectively clustered and provide a general description for each cluster. Detailed material retrieval is not necessary at this stage.

Please present the output in the following format:
"[1,2,3,4]: all bedding on the bed, including pillow and bedsheet"

Ensure that:
	1.	Each cluster represents a coherent grouping based on segment similarity.
	2.	Unique segments are separated where applicable.
	3.	Each segment appears only once in a group.
    4. output the results only with no other information.

"""
"""
Segment grouping prompt for clustering similar segments.

Args (format):
    descriptions (str): Formatted descriptions of segmented parts
"""


# =============================================================================
# Color Extraction Prompts
# =============================================================================

COLOR_INFORMATION_EXTRACTION = """The top row in this image displays photos captured from an indoor environment, cropped to focus on showing a {} in a real room and please disregard any background elements in the photos. The second row contains a series of 3D renderings and corresponding segmentation masks, with the interested region highlighted in red, and it is described as: {}. Although the furniture in the renderings is not an exact match to the photographed object, it is very similar, allowing for effective material matching. Our goal is to assign materials to the red-highligted part of the 3D mesh, ensuring visual alignment with the object shown in the top photos. Please describe the color of the highlighted part based on the corresponding part in the top image. If the highlighted part is not visible in the top images, suggest the most suitable color to make the lower furniture resemble the top furniture as closely as possible. Provide the RGB values so I can reproduce it, using the best RGB approximation available. Output only the RGB values directly in the form [R, G, B], without any extra text"""
"""
Color extraction prompt for RGB value estimation.

Args (format):
    type_str (str): Object type
    description (str): Description of the red-highlighted segment
"""


# =============================================================================
# Shape Analysis Prompts
# =============================================================================

PROMPT_SHAPE = "Using its visual features (such as color, pattern, and structure), along with your knowledge of materials, identify and describe the parts of this object and infer the materials of each. Provide a concise description in no more than 50 words, including the type of object, its parts, and the material composition of each. Color and Material are the most important features to consider. "
"""
Shape and material analysis prompt for initial object understanding.
"""


# =============================================================================
# Crop Selection Prompts
# =============================================================================

QUERY_CROPPING = """

This image has two sections. The top section is a real-life photo of a {}. Several black bounding boxes indicate cropped material areas in this image, each labeled with a white index on a black background. Below the photo is a 3D rendering of a retrieved model. The focus area is highlighted in red, which was described as {}. Although the 3D model does not exactly match the furniture in the image, it closely resembles it. The task is to find a material for the red-highlighted part, so the retrieved model can be rendered to resemble the real-life item in the top image. Using both visual information and rendering knowledge, please identify if any of the cropped material areas correspond to the red-highlighted part. If a match exists, directly output the index number of the corresponding patch. In some cases, the segmented part might not be present in the top image but may have a similar material to other parts. This is acceptable, do your best to give me a prediction. If finding a match is too difficult, simply output None. Direct output the index number or None without any other words or punctuation.

"""
"""
Crop selection prompt for matching material areas to segmented parts.

Args (format):
    type_str (str): Object type
    description (str): Description of the red-highlighted segment
"""


# =============================================================================
# Helper Functions
# =============================================================================

def get_object_intro_prompt(type_str: str) -> str:
    """
    Get the single object introduction prompt.

    Args:
        type_str: Object type (e.g., "Chair", "Table")

    Returns:
        Formatted introduction prompt string
    """
    return ADD_PROMPT.format(type_str, type_str)


def get_dual_section_intro_prompt(type_str: str, num_parts: int, parts_list: str) -> str:
    """
    Get the dual-section introduction prompt for combined photo + mesh images.

    Args:
        type_str: Object type
        num_parts: Number of parts in the mesh
        parts_list: Comma-separated list of part names

    Returns:
        Formatted dual-section prompt string
    """
    return ADD_PROMPT_DOUBLE.format(type_str, type_str, type_str, num_parts, parts_list)


def get_introduction_prompt(type_str: str, description: str) -> str:
    """
    Get the two-image context introduction prompt.

    Args:
        type_str: Object type
        description: Description of the highlighted segment

    Returns:
        Formatted introduction prompt string
    """
    return INTRODUCTION_PROMPT.format(type_str, description)


def get_segmentation_prompt(type_str: str) -> str:
    """
    Get the segmentation description prompt.

    Args:
        type_str: Object type

    Returns:
        Formatted segmentation prompt string
    """
    return SEGMENTATION_PROMPT.format(type_str)


def get_segment_group_prompt(descriptions: str) -> str:
    """
    Get the segment grouping prompt.

    Args:
        descriptions: Formatted descriptions of segmented parts

    Returns:
        Formatted segment grouping prompt string
    """
    return SEGMENT_GROUP_PROMPT.format(descriptions)


def get_color_extraction_prompt(type_str: str, description: str) -> str:
    """
    Get the color extraction prompt for RGB estimation.

    Args:
        type_str: Object type
        description: Description of the highlighted segment

    Returns:
        Formatted color extraction prompt string
    """
    return COLOR_INFORMATION_EXTRACTION.format(type_str, description)


def get_crop_selection_prompt(type_str: str, description: str) -> str:
    """
    Get the crop selection prompt.

    Args:
        type_str: Object type
        description: Description of the highlighted segment

    Returns:
        Formatted crop selection prompt string
    """
    return QUERY_CROPPING.format(type_str, description)
