#!/usr/bin/env python3
"""
Download example scans from HuggingFace for LiteReality.

This script downloads the example RGB-D scans and places them in the correct
location (./scans) relative to the project root.

Usage:
    python litereality/utils/download_example_scans.py

After download, run batch_test.sh to test on all example scans:
    bash batch_test.sh
"""

import os
import sys
import zipfile
from pathlib import Path
from huggingface_hub import snapshot_download


def get_project_root():
    """Get the project root directory."""
    # This script is in litereality/utils/, so go up 2 levels
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    return project_root


def extract_and_remove_zips(directory: Path):
    """
    Extract all zip files in the directory (recursively) and remove them.
    
    Args:
        directory: Directory to search for zip files
    """
    zip_files = list(directory.rglob("*.zip"))
    
    if not zip_files:
        return
    
    print(f"\nFound {len(zip_files)} zip file(s) to extract...")
    
    for zip_path in zip_files:
        print(f"  Extracting: {zip_path.name}")
        try:
            # Extract to the same directory as the zip file
            extract_dir = zip_path.parent
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            # Remove the zip file after successful extraction
            zip_path.unlink()
            print(f"    ✓ Extracted and removed: {zip_path.name}")
        except Exception as e:
            print(f"    ✗ Error extracting {zip_path.name}: {e}")
            # Don't remove if extraction failed


def download_example_scans(output_dir: str = None):
    """
    Download example scans from HuggingFace.

    Args:
        output_dir: Directory to download scans to. If None, uses ./scans
                   relative to project root.
    """
    project_root = get_project_root()

    if output_dir is None:
        output_dir = project_root / "scans"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("LiteReality - Download Example Scans")
    print("=" * 60)
    print(f"\nRepository: zhening/LiteReality_example_scans")
    print(f"Target directory: {output_dir.absolute()}")
    print()

    snapshot_download(
        repo_id="zhening/LiteReality_example_scans",
        repo_type="dataset",
        local_dir=str(output_dir)
    )

    print(f"\n{'=' * 60}")
    print("Download complete!")
    print(f"{'=' * 60}")
    
    # Extract zip files and remove them
    extract_and_remove_zips(output_dir)
    
    print(f"\nExample scans available at: {output_dir.absolute()}")
    print("\nAvailable scans:")

    # List downloaded scans
    scans = sorted([d for d in output_dir.iterdir() if d.is_dir() and not d.name.startswith('.')])
    for scan in scans:
        print(f"  - {scan.name}")

    print(f"\nTo test on all example scans, run:")
    print(f"  bash batch_test.sh")
    print()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Download LiteReality example scans from HuggingFace"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: ./scans relative to project root)"
    )

    args = parser.parse_args()
    download_example_scans(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
