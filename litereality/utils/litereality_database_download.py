import os
import subprocess
import tarfile
from pathlib import Path
from huggingface_hub import snapshot_download
from tqdm import tqdm
import shutil

REPO_ID = "zhening/LiteReality-DataBase"
BASE_DIR = Path("./litereality_database")
# The folder created after PBR extraction (adjust name if different in the archive)
PBR_EXTRACT_PATH = BASE_DIR / "PBR_materials" 

def run_setup():
    # 1. DOWNLOAD
    # Check if the directory exists and isn't empty
    if BASE_DIR.exists() and any(BASE_DIR.iterdir()):
        print(f"--- 1/3: {BASE_DIR} is not empty. Skipping/Verifying download ---")
    else:
        print(f"--- 1/3: Downloading from {REPO_ID} ---")
        BASE_DIR.mkdir(parents=True, exist_ok=True)
    
    # snapshot_download naturally handles resume/caching, so it's safe to call
    snapshot_download(repo_id=REPO_ID, repo_type="dataset", local_dir=str(BASE_DIR))


    split_parts = sorted(list(BASE_DIR.glob("PBR_materials_part_*.tar")))
    combined_pbr = BASE_DIR / "PBR_materials_TOTAL.tar"

    # Only stitch if the PBR folder doesn't exist and we have parts to stitch
    if PBR_EXTRACT_PATH.exists():
        print(f"--- 2/3: PBR materials already extracted in {PBR_EXTRACT_PATH}. Skipping. ---")
    elif split_parts:
        total_size = sum(p.stat().st_size for p in split_parts)
        print(f"--- 2/3: Stitching PBR (Total: {total_size / 1e9:.2f} GB) ---")
        
        if not combined_pbr.exists():
            pv_installed = shutil.which("pv") is not None
            part_names = " ".join([f"'{str(p)}'" for p in split_parts])

            if pv_installed:
                print("Using 'pv' for high-speed system stitching...")
                cmd = f"cat {part_names} | pv -s {total_size} > '{combined_pbr}'"
                subprocess.run(cmd, shell=True, check=True)
            else:
                print("Using Python with 128MB buffer...")
                with open(combined_pbr, 'wb') as master:
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc="Stitching") as pbar:
                        for part in split_parts:
                            with open(part, 'rb') as f_in:
                                while True:
                                    chunk = f_in.read(128 * 1024 * 1024)
                                    if not chunk: break
                                    master.write(chunk)
                                    pbar.update(len(chunk))

        print("--- Extracting PBR (Multi-core) ---")
        subprocess.run(f"tar -xf '{combined_pbr}' -C '{BASE_DIR}'", shell=True, check=True)
        
        # Cleanup only after successful extraction
        combined_pbr.unlink()
        for p in split_parts: p.unlink()

    # 3. REMAINING ASSETS
    print("--- 3/3: Extracting remaining assets ---")
    for arch in BASE_DIR.glob("*.tar*"):
        # Skip if it's the PBR parts (already handled) or if extraction folder exists
        if arch.is_file() and "TOTAL" not in arch.name and "PBR_materials" not in arch.name:
            # Simple check: if a folder with the same name as the tar (minus extension) exists, skip
            folder_name = arch.stem # e.g., 'assets' from 'assets.tar'
            if (BASE_DIR / folder_name).exists():
                print(f"Skipping {arch.name}, folder '{folder_name}' already exists.")
                # Optional: arch.unlink() if you want to clean up leftovers
                continue

            print(f"Extracting {arch.name}...")
            subprocess.run(f"tar -xf '{arch}' -C '{BASE_DIR}'", shell=True)
            arch.unlink()

    print("\nDONE: Database Ready.")

if __name__ == "__main__":
    run_setup()