"""
Qwen3-VL wrapper utilities for litereality material painting module.

This module provides Qwen3-VL-based replacements for OpenAI API calls,
following the same pattern as litereality/LR_retrieval/utils.py
"""

# Standard library imports
import json
import os
import sys
import time
from pathlib import Path

# Add path to import qwan module from LR_retrieval
current_dir = Path(__file__).parent
lr_retrieval_path = current_dir.parent.parent.parent / "litereality" / "LR_retrieval"
sys.path.insert(0, str(lr_retrieval_path))

# Import Qwen3-VL functions
try:
    from qwan import load_model as qwen_load_model, inference as qwen_inference, unload_model as qwen_unload_model
    # Import inference_multi directly from qwan module
    import qwan
    qwen_inference_multi = qwan.inference_multi
    QWEN_AVAILABLE = True
except (ImportError, AttributeError) as e:
    QWEN_AVAILABLE = False
    print(f"Warning: Qwen3-VL not available: {e}")
    print("Please ensure qwan.py is accessible from litereality/LR_retrieval/")
    qwen_inference_multi = None


def process_image_qwen(image_path, prompt, max_new_tokens=300, max_image_size=800):
    """
    Process image with Qwen3-VL model (replacement for OpenAI process_image).
    
    Args:
        image_path: Path to the image file
        prompt: Text prompt/question about the image
        max_new_tokens: Maximum tokens to generate (default: 300)
        max_image_size: Maximum image dimension (default: 800)
    
    Returns:
        dict: Response in OpenAI-compatible format for backward compatibility
    """
    if not QWEN_AVAILABLE:
        raise RuntimeError("Qwen3-VL not available. Please ensure qwan.py is accessible.")
    
    try:
        # Use Qwen inference
        message = qwen_inference(image_path, prompt, max_new_tokens=max_new_tokens, max_image_size=max_image_size)
        
        # Return in OpenAI-compatible format for backward compatibility
        return {
            'choices': [{
                'message': {
                    'content': message
                }
            }]
        }
    except Exception as e:
        print(f"Error in Qwen inference: {e}")
        raise


def multi_try_ensure_response_qwen(image_path, prompt, save_folder, max_tries=5, Force=False, max_new_tokens=300, max_image_size=800):
    """
    Qwen3-VL version of multi_try_ensure_response with caching support.
    
    Args:
        image_path: Path to the image file
        prompt: Text prompt/question about the image
        save_folder: Folder to save response files
        max_tries: Maximum number of retry attempts (default: 5)
        Force: If True, ignore cached responses (default: False)
        max_new_tokens: Maximum tokens to generate (default: 300)
        max_image_size: Maximum image dimension (default: 800)
    
    Returns:
        str: Generated response text, or None if failed
    """
    if not QWEN_AVAILABLE:
        raise RuntimeError("Qwen3-VL not available.")
    
    os.makedirs(save_folder, exist_ok=True)
    save_json_path = os.path.join(save_folder, "response.json")
    save_txt_path = os.path.join(save_folder, "response.txt")
    
    # Check cache if not forcing
    if not Force and os.path.exists(save_json_path) and os.path.getsize(save_json_path) > 0:
        with open(save_json_path, "r") as f:
            try:
                response = json.load(f)
                if 'choices' in response and len(response['choices']) > 0:
                    message = response['choices'][0]['message']['content']
                elif 'message' in response:
                    message = response['message']
                elif 'content' in response:
                    message = response['content']
                else:
                    message = str(response)
                
                print(f"Using cached response: {message[:50]}...")
                # Save message to text file for consistency
                with open(save_txt_path, "w") as txt_file:
                    txt_file.write(message)
                return message
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Invalid cache file, proceeding with new request... ({e})")
    
    # Try inference with retries
    count = 0
    message = None
    
    while count < max_tries:
        try:
            print(f"Qwen3-VL inference attempt {count + 1}/{max_tries} for {image_path}")
            import torch
            torch.cuda.empty_cache()
            
            # Adjust image size for large images
            try:
                from PIL import Image
                img = Image.open(image_path)
                img_size = max(img.size)
                img.close()
                actual_max_size = max_image_size
                if img_size > 2000:
                    actual_max_size = 600
                elif img_size > 1500:
                    actual_max_size = 700
            except:
                actual_max_size = max_image_size
            
            response = process_image_qwen(image_path, prompt, max_new_tokens=max_new_tokens, max_image_size=actual_max_size)
            message = response['choices'][0]['message']['content'].strip()
            
            if message and len(message) > 0:
                # Save response
                with open(save_json_path, "w") as f:
                    json.dump(response, f, indent=4)
                with open(save_txt_path, "w") as f:
                    f.write(message)
                print(f"Success: query {image_path}")
                return message
            else:
                print("Empty response received, retrying...")
                count += 1
                time.sleep(2)
                
        except Exception as e:
            print(f"Error on attempt {count + 1}: {e}")
            count += 1
            if count < max_tries:
                time.sleep(2)
            else:
                print(f"Failed after {max_tries} attempts")
                return None
    
    return None


def process_image_multi_qwen(image_paths, prompt, max_new_tokens=300, max_image_size=800):
    """
    Process multiple images with Qwen3-VL model (replacement for OpenAI multi-image API).
    
    Args:
        image_paths: List of paths to image files
        prompt: Text prompt/question about the images
        max_new_tokens: Maximum tokens to generate (default: 300)
        max_image_size: Maximum image dimension (default: 800)
    
    Returns:
        dict: Response in OpenAI-compatible format for backward compatibility
    """
    if not QWEN_AVAILABLE:
        raise RuntimeError("Qwen3-VL not available. Please ensure qwan.py is accessible.")
    
    if not isinstance(image_paths, list) or len(image_paths) == 0:
        raise ValueError("image_paths must be a non-empty list")
    
    try:
        # Use Qwen multi-image inference
        message = qwen_inference_multi(image_paths, prompt, max_new_tokens=max_new_tokens, max_image_size=max_image_size)
        
        # Return in OpenAI-compatible format for backward compatibility
        return {
            'choices': [{
                'message': {
                    'content': message
                }
            }]
        }
    except Exception as e:
        print(f"Error in Qwen multi-image inference: {e}")
        raise

