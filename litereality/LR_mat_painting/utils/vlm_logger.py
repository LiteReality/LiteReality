"""
VLM Conversation Logger

Centralized logging system for all VLM (Vision-Language Model) interactions.
Captures complete conversation details including prompts, input images, responses,
and parsed results for debugging and visualization.
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any


def log_vlm_conversation(
    scene_path: str,
    phase: str,
    part_id: str,
    object_name: str,
    prompt: str,
    input_images: List[Dict[str, Any]],
    raw_response: Dict[str, Any],
    parsed_result: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    prompt_template: Optional[str] = None,
    parsing_errors: Optional[List[str]] = None,
    success: bool = True,
    error_message: Optional[str] = None
) -> str:
    """
    Log a complete VLM conversation with all details.
    
    Args:
        scene_path: Path to scene folder
        phase: Phase name (e.g., "phase_1_text_visual_selection")
        part_id: Part identifier (e.g., "solid_001")
        object_name: Object name (e.g., "Chair")
        prompt: Full prompt text sent to VLM
        input_images: List of input image dictionaries with paths, descriptions, etc.
        raw_response: Raw VLM response dictionary
        parsed_result: Parsed result dictionary (optional)
        metadata: Additional metadata dictionary (optional)
        prompt_template: Template name if applicable (optional)
        parsing_errors: List of parsing errors if any (optional)
        success: Whether the query was successful (default: True)
        error_message: Error message if query failed (optional)
    
    Returns:
        str: Path to the saved log file
    """
    # Create log directory structure
    object_name_clean = ''.join(filter(lambda x: not x.isdigit(), object_name))
    gpt_folder = f"{scene_path}/{object_name_clean}_gpt"
    log_dir = f"{gpt_folder}/vlm_conversations"
    os.makedirs(log_dir, exist_ok=True)
    
    # Calculate processing time if available in response
    processing_time = None
    if isinstance(raw_response, dict) and "processing_time" in raw_response:
        processing_time = raw_response["processing_time"]
    
    # Build conversation log structure
    conversation_log = {
        "timestamp": datetime.now().isoformat(),
        "phase": phase,
        "part_id": part_id,
        "object_name": object_name,
        "vlm_query": {
            "prompt": prompt,
            "prompt_template": prompt_template,
            "input_images": input_images,
            "raw_response": raw_response,
            "parsed_result": parsed_result,
            "parsing_errors": parsing_errors or [],
            "processing_time_seconds": processing_time,
            "success": success,
            "error_message": error_message
        },
        "metadata": metadata or {}
    }
    
    # Save log file
    log_filename = f"{phase}_part_{part_id}.json"
    log_path = os.path.join(log_dir, log_filename)
    
    with open(log_path, 'w') as f:
        json.dump(conversation_log, f, indent=2)
    
    # Update index file
    update_conversation_index(scene_path, object_name_clean, phase, part_id, log_path)
    
    return log_path


def update_conversation_index(scene_path: str, object_name: str, phase: str, part_id: str, log_path: str):
    """
    Update the conversation index file with new log entry.
    
    Args:
        scene_path: Path to scene folder
        object_name: Object name (cleaned)
        phase: Phase name
        part_id: Part identifier
        log_path: Path to the log file
    """
    gpt_folder = f"{scene_path}/{object_name}_gpt"
    log_dir = f"{gpt_folder}/vlm_conversations"
    index_path = os.path.join(log_dir, "vlm_conversations_index.json")
    
    # Load existing index or create new one
    if os.path.exists(index_path):
        with open(index_path, 'r') as f:
            index = json.load(f)
    else:
        index = {
            "object_name": object_name,
            "scene_path": scene_path,
            "conversations": []
        }
    
    # Add or update entry
    entry = {
        "phase": phase,
        "part_id": part_id,
        "log_path": log_path,
        "relative_path": os.path.relpath(log_path, scene_path),
        "timestamp": datetime.now().isoformat()
    }
    
    # Remove existing entry if present
    index["conversations"] = [
        conv for conv in index["conversations"]
        if not (conv["phase"] == phase and conv["part_id"] == part_id)
    ]
    
    # Add new entry
    index["conversations"].append(entry)
    
    # Save index
    with open(index_path, 'w') as f:
        json.dump(index, f, indent=2)


def get_vlm_conversation(scene_path: str, object_name: str, phase: str, part_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve a VLM conversation log.
    
    Args:
        scene_path: Path to scene folder
        object_name: Object name (cleaned)
        phase: Phase name
        part_id: Part identifier
    
    Returns:
        dict: Conversation log dictionary or None if not found
    """
    object_name_clean = ''.join(filter(lambda x: not x.isdigit(), object_name))
    gpt_folder = f"{scene_path}/{object_name_clean}_gpt"
    log_dir = f"{gpt_folder}/vlm_conversations"
    log_filename = f"{phase}_part_{part_id}.json"
    log_path = os.path.join(log_dir, log_filename)
    
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            return json.load(f)
    return None


def get_all_conversations(scene_path: str, object_name: str) -> List[Dict[str, Any]]:
    """
    Get all VLM conversations for an object.
    
    Args:
        scene_path: Path to scene folder
        object_name: Object name (cleaned)
    
    Returns:
        list: List of conversation log dictionaries
    """
    object_name_clean = ''.join(filter(lambda x: not x.isdigit(), object_name))
    gpt_folder = f"{scene_path}/{object_name_clean}_gpt"
    log_dir = f"{gpt_folder}/vlm_conversations"
    index_path = os.path.join(log_dir, "vlm_conversations_index.json")
    
    if not os.path.exists(index_path):
        return []
    
    with open(index_path, 'r') as f:
        index = json.load(f)
    
    conversations = []
    for entry in index.get("conversations", []):
        log_path = entry.get("log_path")
        if log_path and os.path.exists(log_path):
            with open(log_path, 'r') as f:
                conversations.append(json.load(f))
    
    return conversations


def prepare_image_info(image_path: str, description: str = "", image_type: str = "unknown") -> Dict[str, Any]:
    """
    Prepare image information dictionary for logging.
    
    Args:
        image_path: Path to image file (can be relative or absolute)
        description: Description of the image
        image_type: Type of image (e.g., "reference_image", "3d_rendering", "albedo_map")
    
    Returns:
        dict: Image information dictionary
    """
    # Convert to absolute path if relative
    if not os.path.isabs(image_path):
        # Assume relative to current working directory
        abs_path = os.path.abspath(image_path)
    else:
        abs_path = image_path
    
    # Get image dimensions if file exists
    width = None
    height = None
    if os.path.exists(abs_path):
        try:
            from PIL import Image
            with Image.open(abs_path) as img:
                width, height = img.size
        except Exception:
            pass
    
    return {
        "path": image_path,
        "absolute_path": abs_path,
        "description": description,
        "image_type": image_type,
        "width": width,
        "height": height
    }
