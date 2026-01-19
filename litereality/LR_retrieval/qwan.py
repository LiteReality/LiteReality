import os
import torch
import sys
sys.modules['torch'] = torch

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image

# Global variables to store loaded model and processor
_model = None
_processor = None

# Default model path - can be overridden via QWEN_MODEL_PATH environment variable
DEFAULT_MODEL_PATH = "third_party/pre-trained/qwen3-vl-8b-instruct"


def load_model(model_path=None):
    """
    Load the Qwen3-VL model and processor into GPU memory.

    Args:
        model_path: Path to the model directory. If None, uses QWEN_MODEL_PATH
                    environment variable or falls back to DEFAULT_MODEL_PATH.

    Returns:
        None (model and processor are stored globally)
    """
    global _model, _processor

    if _model is not None:
        return

    # Determine model path: argument > env var > default
    if model_path is None:
        model_path = os.environ.get("QWEN_MODEL_PATH", DEFAULT_MODEL_PATH)

    print(f"    Loading model from: {model_path}")
    
    # Load processor
    print(f"    Loading processor...")
    _processor = AutoProcessor.from_pretrained(model_path)
    
    # Load model
    print(f"    Loading model to GPU...")
    _model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, 
        dtype="auto", 
        device_map="auto",
        trust_remote_code=True
    )


def inference(image_path, prompt, max_new_tokens=256, max_image_size=448):
    """
    Perform inference with the loaded model.

    Args:
        image_path: Path to image file or URL
        prompt: Text prompt/question about the image
        max_new_tokens: Maximum number of tokens to generate (default: 256)
        max_image_size: Maximum image dimension to resize to (default: 448)

    Returns:
        str: Generated response text
    """
    global _model, _processor

    if _model is None or _processor is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    
    # Load and preprocess image
    if image_path.startswith("http"):
        image = image_path
    else:
        image = Image.open(image_path)
        
        # Resize large images to avoid CUDA errors
        if max(image.size) > max_image_size:
            ratio = max_image_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            # Ensure dimensions are multiples of 14 (patch size)
            new_size = (new_size[0] - (new_size[0] % 14), new_size[1] - (new_size[1] % 14))
            # Ensure minimum size
            if new_size[0] < 14:
                new_size = (14, new_size[1])
            if new_size[1] < 14:
                new_size = (new_size[0], 14)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
    
    # Prepare messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    
    # Process inputs
    inputs = _processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = {k: v.to(_model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
    # Generate response
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    try:
        with torch.inference_mode():
            generated_ids = _model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=_processor.tokenizer.eos_token_id,
                use_cache=True
            )
    except RuntimeError as e:
        error_str = str(e)
        if "CUDA" in error_str or "out of memory" in error_str.lower() or "driver" in error_str.lower():
            print(f"⚠️ CUDA error occurred, retrying with smaller max_new_tokens...")
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Try with much smaller max_new_tokens
            retry_max_tokens = max(max_new_tokens // 3, 64)
            
            try:
                with torch.inference_mode():
                    generated_ids = _model.generate(
                        **inputs, 
                        max_new_tokens=retry_max_tokens,
                        do_sample=False,
                        pad_token_id=_processor.tokenizer.eos_token_id,
                        use_cache=True
                    )
            except RuntimeError as e2:
                # If still fails, try even smaller
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                with torch.inference_mode():
                    generated_ids = _model.generate(
                        **inputs, 
                        max_new_tokens=64,
                        do_sample=False,
                        pad_token_id=_processor.tokenizer.eos_token_id,
                        use_cache=False  # Disable cache to save memory
                    )
        else:
            raise
    
    # Decode output
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
    ]
    output_text = _processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text[0]


def inference_multi(image_paths, prompt, max_new_tokens=256, max_image_size=448):
    """
    Perform inference with multiple images.
    
    Args:
        image_paths: List of image file paths (first is stitched/context, rest are candidates)
        prompt: Text prompt/question about the images
        max_new_tokens: Maximum number of tokens to generate (default: 256)
        max_image_size: Maximum image dimension to resize to (default: 448)
    
    Returns:
        str: Generated response text
    """
    global _model, _processor
    
    if _model is None or _processor is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    
    if not isinstance(image_paths, list) or len(image_paths) == 0:
        raise ValueError("image_paths must be a non-empty list")
    
    # Load and preprocess all images
    images = []
    for image_path in image_paths:
        if image_path.startswith("http"):
            images.append(image_path)
        else:
            image = Image.open(image_path)
            
            # Resize large images to avoid CUDA errors
            if max(image.size) > max_image_size:
                ratio = max_image_size / max(image.size)
                new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                # Ensure dimensions are multiples of 14 (patch size)
                new_size = (new_size[0] - (new_size[0] % 14), new_size[1] - (new_size[1] % 14))
                # Ensure minimum size
                if new_size[0] < 14:
                    new_size = (14, new_size[1])
                if new_size[1] < 14:
                    new_size = (new_size[0], 14)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            images.append(image)
    
    # Prepare messages with multiple images
    content = []
    # Add all images first
    for image in images:
        content.append({"type": "image", "image": image})
    # Add text prompt at the end
    content.append({"type": "text", "text": prompt})
    
    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]
    
    # Process inputs
    inputs = _processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = {k: v.to(_model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
    # Generate response
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    try:
        with torch.inference_mode():
            generated_ids = _model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=_processor.tokenizer.eos_token_id,
                use_cache=True
            )
    except RuntimeError as e:
        error_str = str(e)
        if "CUDA" in error_str or "out of memory" in error_str.lower() or "driver" in error_str.lower():
            print(f"⚠️ CUDA error occurred, retrying with smaller max_new_tokens...")
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Try with much smaller max_new_tokens
            retry_max_tokens = max(max_new_tokens // 3, 64)
            
            try:
                with torch.inference_mode():
                    generated_ids = _model.generate(
                        **inputs, 
                        max_new_tokens=retry_max_tokens,
                        do_sample=False,
                        pad_token_id=_processor.tokenizer.eos_token_id,
                        use_cache=True
                    )
            except RuntimeError as e2:
                # If still fails, try even smaller
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                with torch.inference_mode():
                    generated_ids = _model.generate(
                        **inputs, 
                        max_new_tokens=64,
                        do_sample=False,
                        pad_token_id=_processor.tokenizer.eos_token_id,
                        use_cache=False  # Disable cache to save memory
                    )
        else:
            raise
    
    # Decode output
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
    ]
    output_text = _processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text[0]


def unload_model():
    """
    Unload the model and processor from memory.
    """
    global _model, _processor
    
    if _model is not None:
        del _model
        _model = None
    
    if _processor is not None:
        del _processor
        _processor = None
    
    torch.cuda.empty_cache()
    print("Model unloaded.")


# Example usage
if __name__ == "__main__":
    # Load model once
    load_model()
    
    # Perform multiple inferences
    image_path = sys.argv[1] if len(sys.argv) > 1 else "cache/scene_parsing_cache/Girton_before.png"
    prompt = "Describe what you see in this image in detail. What objects or scenes are visible?"
    
    print(f"\nUsing image: {image_path}")
    print(f"Prompt: {prompt}\n")
    
    import time
    start_time = time.time()
    result = inference(image_path, prompt)

    prompt_2 = "what movie do this think this is from?"

    result = inference(image_path, prompt_2)
    elapsed = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"Response Time: {elapsed:.2f}s")
    print(f"{'='*60}")
    print("Response:")
    print("-"*60)
    print(result)
    print("-"*60)
    
    # Unload when done (optional)
    unload_model()
