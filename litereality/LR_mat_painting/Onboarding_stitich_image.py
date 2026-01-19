"""
Image Stitching for LiteReality Material Pipeline

This script performs Step 1 of the material painting pipeline:
- Stitches multiple captured frame images into a single overview image
- Creates stitched_image.jpg from all frame_*.jpg images in captured_images/
"""

import os
import argparse
from PIL import Image, ImageOps


def get_images(folder_path, prefix="frame"):
    """Retrieve all images starting with a specific prefix from a folder."""
    print(f"Retrieving images with prefix '{prefix}' from {folder_path}...")
    images = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.startswith(prefix) and filename.endswith(('.png', '.jpg', '.jpeg')):
            images.append(Image.open(os.path.join(folder_path, filename)))
    print(f"Found {len(images)} images.")
    return images


def add_padding(image, target_width, target_height):
    """Add padding to center the image in target dimensions."""
    return ImageOps.pad(image, (target_width, target_height), color=(255, 255, 255))


def stitch_images(images, output_path="stitched_image.jpg"):
    """Stitch multiple images into a grid layout."""
    print(f"Stitching {len(images)} images...")
    num_images = len(images)

    # Determine layout based on the number of images
    if num_images == 2:
        cols, rows = 2, 1
    elif num_images == 3:
        cols, rows = 3, 1
    elif num_images == 4:
        cols, rows = 2, 2
    elif num_images == 5:
        cols, rows = 3, 2
    else:
        cols, rows = 3, 2

    print(f"Using layout: {cols}x{rows}")

    # Determine max width and height of each image to maintain ratio and padding consistency
    max_width = max(image.width for image in images)
    max_height = max(image.height for image in images)

    # Calculate the full dimensions for the output image
    stitched_width = cols * max_width
    stitched_height = rows * max_height

    # Create a blank canvas with the full stitched size
    stitched_image = Image.new("RGB", (stitched_width, stitched_height), (255, 255, 255))

    # Place each image onto the canvas with padding if needed
    for index, image in enumerate(images):
        row = index // cols
        col = index % cols
        # Add padding to the image if it's smaller than max dimensions
        padded_image = add_padding(image, max_width, max_height)
        # Paste the image onto the stitched image at the calculated position
        x = col * max_width
        y = row * max_height
        stitched_image.paste(padded_image, (x, y))

    # Save the final stitched image
    stitched_image.save(output_path)
    print(f"Stitched image saved to {output_path}")


def stitch_capture_images(folder_path):
    """Stitch all frame images in a folder into a single image."""
    print(f"\n[PROCESS] Stitching capture images from {folder_path}...")
    images = get_images(folder_path)
    if images:
        output_path = f"{folder_path}/stitched_image.jpg"
        stitch_images(images, output_path=output_path)
        print(f"✓ Successfully created stitched image: {output_path}")
    else:
        print("✗ No images found starting with 'frame'. Stitching skipped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stitch captured images for LiteReality material pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--scene",
        type=str,
        required=True,
        help="Path to the object folder containing captured_images/."
    )

    args = parser.parse_args()
    scene_path = args.scene

    print(f"\n{'='*60}")
    print(f"🧩 LITE REALITY - IMAGE STITCHING")
    print(f"Processing: {scene_path}")
    print(f"{'='*60}")

    # Main processing: stitch captured images
    captured_path = f"{scene_path}/captured_images"
    stitch_capture_images(captured_path)

    print(f"\n{'='*60}")
    print("✓ Image stitching completed successfully!")
    print(f"{'='*60}")
