import subprocess
from pathlib import Path
from huggingface_hub import snapshot_download
import shutil

REPO_ID = "zhening/LiteReality-DataBase"
BASE_DIR = Path("./litereality_database")
# The folder created after PBR extraction (adjust name if different in the archive)
PBR_EXTRACT_PATH = BASE_DIR / "PBR_materials" 

def run_setup():
    # 1. DOWNLOAD
    BASE_DIR.mkdir(parents=True, exist_ok=True)

    # Check if download is already complete by looking for expected files
    split_parts = sorted(list(BASE_DIR.glob("PBR_materials_part_*.tar")))
    other_archives = list(BASE_DIR.glob("*.tar*"))
    pbr_extracted = (BASE_DIR / "PBR_materials").exists()

    # Skip download if we have the split parts OR the PBR folder already exists
    if split_parts or pbr_extracted or other_archives:
        print(f"--- 1/3: Download already complete in {BASE_DIR}. Skipping. ---")
    else:
        print(f"--- 1/3: Downloading from {REPO_ID} ---")
        snapshot_download(repo_id=REPO_ID, repo_type="dataset", local_dir=str(BASE_DIR))


    # Refresh split_parts list after potential download
    split_parts = sorted(list(BASE_DIR.glob("PBR_materials_part_*.tar")))

    # Only extract if the PBR folder doesn't exist and we have parts
    if PBR_EXTRACT_PATH.exists():
        print(f"--- 2/3: PBR materials already extracted in {PBR_EXTRACT_PATH}. Skipping. ---")
    elif split_parts:
        total_size = sum(p.stat().st_size for p in split_parts)
        print(f"--- 2/3: Extracting PBR directly (Total: {total_size / 1e9:.2f} GB) ---")

        pv_installed = shutil.which("pv") is not None
        part_names = " ".join([f"'{str(p)}'" for p in split_parts])

        if pv_installed:
            # Stream directly to tar with progress - no intermediate file needed
            print("Using 'pv' for progress (streaming directly to tar)...")
            cmd = f"cat {part_names} | pv -s {total_size} | tar -xf - -C '{BASE_DIR}'"
        else:
            # Stream directly to tar without progress
            print("Streaming directly to tar (install 'pv' for progress bar)...")
            cmd = f"cat {part_names} | tar -xf - -C '{BASE_DIR}'"

        subprocess.run(cmd, shell=True, check=True)

        # Cleanup split parts after successful extraction
        for p in split_parts:
            p.unlink()

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