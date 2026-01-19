import cv2
import glob
import os
import sys
import argparse
from tqdm import tqdm
import contextlib

# Context manager to suppress stderr (for libpng warnings)
@contextlib.contextmanager
def suppress_stderr():
    """Temporarily suppress stderr to hide libpng warnings"""
    old_stderr = sys.stderr
    try:
        sys.stderr = open(os.devnull, 'w')
        yield
    finally:
        sys.stderr.close()
        sys.stderr = old_stderr


def generate_comparison_video(scene_name: str, fps: int = 5):
    render_path = f"output/whole_scene_render/rendered_rgbd/{scene_name}"
    gt_path = f"input/rgbd/{scene_name}/image/"
    video_path = f"output/whole_scene_render/videos/{scene_name}.mp4"

    # Check if RGBD data exists
    if not os.path.exists(gt_path):
        # List available scenes
        rgbd_base = "input/rgbd"
        available_scenes = []
        if os.path.exists(rgbd_base):
            available_scenes = [d for d in os.listdir(rgbd_base) 
                              if os.path.isdir(os.path.join(rgbd_base, d))]
        
        error_msg = (
            f"RGBD data directory not found: {gt_path}\n"
            f"Please ensure preprocessing has been completed for scene '{scene_name}'.\n"
        )
        if available_scenes:
            error_msg += f"\nAvailable scenes: {', '.join(available_scenes)}"
        raise FileNotFoundError(error_msg)

    total_num_images = len(glob.glob(os.path.join(gt_path, "frame_*.jpg")))
    if total_num_images == 0:
        raise FileNotFoundError(
            f"No images found in {gt_path}\n"
            f"Please ensure preprocessing has been completed for scene '{scene_name}'."
        )

    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Check if paths exist
    if not os.path.exists(render_path):
        raise FileNotFoundError(f"Rendered images directory not found: {render_path}\n"
                              f"Please run the whole scene rendering step first (Step 4 in script.sh).")
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Ground truth images directory not found: {gt_path}")
    
    # Load first images to determine dimensions
    gt_file = os.path.join(gt_path, "frame_0.jpg")
    render_file = os.path.join(render_path, "render_image_0.png")

    if not os.path.exists(gt_file):
        raise FileNotFoundError(f"Ground truth image not found: {gt_file}")
    if not os.path.exists(render_file):
        raise FileNotFoundError(f"Rendered image not found: {render_file}\n"
                              f"Please run the whole scene rendering step first (Step 4 in script.sh).")

    # Suppress libpng EXIF warnings when loading images
    with suppress_stderr():
        gt = cv2.imread(gt_file)
        render = cv2.imread(render_file)

    if gt is None or render is None:
        raise ValueError(f"Error: Failed to read images.\n"
                         f"GT file: {gt_file}\n"
                         f"Render file: {render_file}")

    gt_resized = cv2.resize(gt, (render.shape[1], render.shape[0]))
    gt_rot = cv2.rotate(gt_resized, cv2.ROTATE_90_CLOCKWISE)
    render_rot = cv2.rotate(render, cv2.ROTATE_90_CLOCKWISE)
    combined = cv2.hconcat([gt_rot, render_rot])
    height, width, _ = combined.shape

    video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    print(f"Generating video for scene: {scene_name}")
    for i in tqdm(range(total_num_images), desc="Processing frames"):
        gt_file = os.path.join(gt_path, f"frame_{i}.jpg")
        render_file = os.path.join(render_path, f"render_image_{i}.png")

        # Suppress libpng EXIF warnings when loading images
        with suppress_stderr():
            gt = cv2.imread(gt_file)
            render = cv2.imread(render_file)

        if gt is None or render is None:
            print(f"Skipping frame {i}: one of the images is missing.")
            continue

        gt_resized = cv2.resize(gt, (render.shape[1], render.shape[0]))
        gt_rot = cv2.rotate(gt_resized, cv2.ROTATE_90_CLOCKWISE)
        render_rot = cv2.rotate(render, cv2.ROTATE_90_CLOCKWISE)
        combined = cv2.hconcat([gt_rot, render_rot])
        video.write(combined)

    video.release()
    print(f"✅ Video saved at: {video_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a comparison video of rendered and ground truth images.")
    parser.add_argument('--name', required=True, help="Name of the scene (e.g., input/object_stage/scene_name)")
    parser.add_argument('--fps', type=int, default=5, help="Frames per second for the output video")

    args = parser.parse_args()
    scene_name = os.path.basename(args.name)
    generate_comparison_video(scene_name, fps=args.fps)