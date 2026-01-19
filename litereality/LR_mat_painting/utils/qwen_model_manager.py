"""
Model manager for Qwen3-VL to handle loading/unloading across multiple scripts.

This ensures the model is loaded once and can be shared across different
functions and scripts in the litereality_mat_painting module.
"""

import sys
import os
from pathlib import Path

# Add path to import qwan module
current_dir = Path(__file__).parent
lr_retrieval_path = current_dir.parent.parent.parent / "litereality" / "LR_retrieval"
sys.path.insert(0, str(lr_retrieval_path))

try:
    from qwan import load_model as qwen_load_model, unload_model as qwen_unload_model
    QWEN_AVAILABLE = True
except ImportError:
    QWEN_AVAILABLE = False
    print("Warning: Qwen3-VL not available.")


_model_loaded = False
_load_count = 0


def ensure_model_loaded():
    """Ensure Qwen model is loaded (idempotent, tracks reference count)"""
    global _model_loaded, _load_count

    if not QWEN_AVAILABLE:
        raise RuntimeError("Qwen3-VL not available. Please ensure qwan.py is accessible.")

    if not _model_loaded:
        print("🤖 Loading Qwen3-VL model...")
        try:
            qwen_load_model()
            _model_loaded = True
            print("✓ Model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load Qwen model: {e}")
            _model_loaded = False  # Make sure it's False
            raise RuntimeError(f"Qwen model loading failed: {e}")

    _load_count += 1
    return _model_loaded


def unload_model_if_loaded():
    """Unload model if it was loaded (decrements reference count)"""
    global _model_loaded, _load_count
    
    if not QWEN_AVAILABLE:
        return
    
    if _load_count > 0:
        _load_count -= 1
    
    # Only unload if no more references
    if _model_loaded and _load_count == 0:
        print("🤖 Unloading Qwen3-VL model...")
        qwen_unload_model()
        _model_loaded = False


def force_unload_model():
    """Force unload model regardless of reference count (use with caution)"""
    global _model_loaded, _load_count

    if not QWEN_AVAILABLE:
        return

    if _model_loaded:
        print("🤖 Force unloading Qwen3-VL model...")
        qwen_unload_model()
        _model_loaded = False
        _load_count = 0


def reset_model_state():
    """Reset model state for debugging (forces reload on next call)"""
    global _model_loaded, _load_count
    _model_loaded = False
    _load_count = 0
    print("🔄 Model state reset - will reload on next call")

