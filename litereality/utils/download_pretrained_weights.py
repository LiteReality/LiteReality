import os
import subprocess
from pathlib import Path
from huggingface_hub import snapshot_download

def setup_workspace():
    # 1. Define and create directories
    base_path = Path.cwd()
    dirs = [
        "third_party/pre-trained",
        "third_party/hf_cache",
        "third_party/GroundingDINO/weights"
    ]
    
    print("--- Creating Directories ---")
    for d in dirs:
        dir_path = base_path / d
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Created: {d}")

    # 2. Set Environment Variable for HF Cache
    os.environ["HF_HOME"] = str(base_path / "third_party/hf_cache")

    # 3. Direct Downloads (GroundingDINO & SAM)
    # Format: (URL, Destination Directory)
    direct_downloads = [
        (
            "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
            "third_party/GroundingDINO/weights/"
        ),
        (
            "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            "third_party/pre-trained/"
        )
    ]

    print("\n--- Downloading Checkpoints (wget) ---")
    for url, dest in direct_downloads:
        filename = url.split('/')[-1]
        target_file = base_path / dest / filename
        
        if target_file.exists():
            print(f"Skipping {filename}, already exists.")
        else:
            print(f"Downloading {filename}...")
            try:
                subprocess.run(["wget", "-q", "--show-progress", url, "-P", str(base_path / dest)], check=True)
            except subprocess.CalledProcessError:
                print(f"Failed to download {filename}. Ensure 'wget' is installed.")

    # 4. HuggingFace Snapshot Downloads
    hf_models = [
        ('facebook/dinov2-base', 'third_party/pre-trained/dinov2-base'),
        ('openai/clip-vit-base-patch32', 'third_party/pre-trained/clip-vit-base-patch32'),
        ('Qwen/Qwen3-VL-8B-Instruct', 'third_party/pre-trained/qwen3-vl-8b-instruct'),
    ]

    print("\n--- Downloading HuggingFace Models ---")
    for repo, local in hf_models:
        print(f"Downloading {repo} to {local}...")
        snapshot_download(
            repo_id=repo, 
            local_dir=str(base_path / local),
            local_dir_use_symlinks=False  # Keeps files in the actual folder
        )

    print("\nSetup complete! All weights are in place.")

if __name__ == "__main__":
    setup_workspace()