#!/usr/bin/env python3
"""
Refactored script for GroundingDINO object retrieval and IOU-based annotation updates.
"""

import json
import re
import argparse
import os
import sys
from pathlib import Path

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# ANSI color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

import cv2
import torch
from torchvision.ops.boxes import box_convert

from third_party.GroundingDINO.groundingdino.util.inference import (
    annotate,
    load_image,
    load_model,
    predict,
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="GroundingDINO object retrieval")
    parser.add_argument(
        "--scene", 
        type=str, 
        default="Trinity_BA",
        help="Scene name (folder in the input/object_stage directory)"
    )
    return parser.parse_args()

# ─────────────────────────── Configuration ────────────────────────────────────
args = parse_args()
INPUT_DIR = Path(f"input/object_stage/{args.scene}")
OUTPUT_DIR = Path("output")
CACHE_DIR = Path("cache")
BBOX_CACHE_DIR = CACHE_DIR / "bbox_correct_cache" / args.scene

MODEL_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../../third_party/GroundingDINO/weights/groundingdino_swint_ogc.pth"
    )
)
CONFIG_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../../third_party/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    )
)

BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.10
IOU_THRESHOLD = 0.2
FALLBACK_THRESHOLD = 0.2  # trigger fallback if max IOU below this
MIN_IOU_SAVE = 0.001     # minimum IOU to keep in fallback mode

# Ensure output directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
BBOX_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def rotate_tensor_90(image: torch.Tensor, clockwise: bool = True) -> torch.Tensor:
    """Rotate a CHW torch.Tensor by 90 degrees."""
    k = 3 if clockwise else 1
    return torch.rot90(image, k=k, dims=(1, 2))


def derive_prompt(name: str) -> str:
    """Generate a language prompt from folder name."""
    prompt = re.sub(r"\d+", "", name)
    prompt = re.sub(r"(?i)wall", "", prompt)
    prompt = prompt.strip("_").replace("_", " ").lower()

    if prompt == "storage":
        synonyms = ["cabinet", "cupboard", "chest", "shelf", "wardrobe"]
        prompt = " . ".join([prompt] + synonyms) + " ."

    return prompt


def load_ground_truth(json_path: Path) -> dict:
    """Load ground truth bounding boxes from JSON."""
    if not json_path.exists():
        return {}
    return json.loads(json_path.read_text())


def compute_iou(box1: list, box2: list) -> float:
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    xa1, ya1, xa2, ya2 = box1
    xb1, yb1, xb2, yb2 = box2
    xi1, yi1 = max(xa1, xb1), max(ya1, yb1)
    xi2, yi2 = min(xa2, xb2), min(ya2, yb2)
    inter = max(0, xi2 - xi1 + 1) * max(0, yi2 - yi1 + 1)
    area1 = max(0, xa2 - xa1 + 1) * max(0, ya2 - ya1 + 1)
    area2 = max(0, xb2 - xb1 + 1) * max(0, yb2 - yb1 + 1)
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def rotate_box_cw_xyxy(box: list, width: int, height: int) -> list:
    """Rotate an [x1,y1,x2,y2] box clockwise around image of size (w,h)."""
    x1, y1, x2, y2 = box
    return [height - 1 - y2, x1, height - 1 - y1, x2]


def rotate_box_ccw_xyxy(box: list, width: int, height: int) -> list:
    """Rotate an [x1,y1,x2,y2] box 90° counter-clockwise around an image of size (width, height)."""
    x1, y1, x2, y2 = box
    
    # Rotate all four corners
    p1 = (y1, width - 1 - x1)
    p2 = (y2, width - 1 - x1)
    p3 = (y1, width - 1 - x2)
    p4 = (y2, width - 1 - x2)
    
    xs = [p[0] for p in [p1, p2, p3, p4]]
    ys = [p[1] for p in [p1, p2, p3, p4]]
    
    return [min(xs), min(ys), max(xs), max(ys)]


def process_folder(folder: Path, model) -> int:
    """Process one folder: predict, compute IOUs, save visuals and updated JSON."""
    name = folder.name
    prompt = derive_prompt(name)
    gt_json = folder / "bbox_info.json"
    gt_data = load_ground_truth(gt_json)

    image_dir = folder / "images"
    image_paths = sorted(image_dir.glob("*.jpg"))
    if not image_paths or not gt_data:
        print(f"{Colors.YELLOW}⚠{Colors.RESET} Skipping {Colors.BOLD}{name}{Colors.RESET}: missing images or ground truth.")
        return 0

    # Prepare log
    log_file = BBOX_CACHE_DIR / f"{name}_iou.log"
    with open(log_file, 'w') as lf:
        lf.write("frame,best_iou,updated\n")

    results = []
    for img_path in image_paths:
        frame = img_path.stem
        orig = cv2.imread(str(img_path))
        if orig is None:
            print(f"{Colors.YELLOW}⚠{Colors.RESET} Warning: failed to load {img_path}")
            continue
        h_orig, w_orig = orig.shape[:2]

        # Load and rotate the model's tensor output
        bgr_raw, tensor_raw = load_image(str(img_path))
        tensor = rotate_tensor_90(tensor_raw)
        bgr = cv2.rotate(orig, cv2.ROTATE_90_CLOCKWISE)
        
        # Convert BGR to RGB for annotate function (annotate expects RGB and converts to BGR internally)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Run inference
        boxes, logits, phrases = predict(
            model=model,
            image=tensor,
            caption=prompt,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
        )

        # Visual base and detections (annotate expects RGB input, returns BGR)
        vis_base = annotate(rgb.copy(), boxes, logits, phrases)
        h_vis, w_vis = vis_base.shape[:2]
        dets = box_convert(
            boxes * torch.tensor([w_vis, h_vis, w_vis, h_vis]),
            in_fmt="cxcywh",
            out_fmt="xyxy",
        ).cpu().int().tolist()

        if frame not in gt_data:
            continue
        gt_box = rotate_box_cw_xyxy(gt_data[frame], w_orig, h_orig)

        # Find best IoU match
        best_iou, best_box = 0.0, None
        for det in dets:
            iou = compute_iou(det, gt_box)
            if iou > best_iou:
                best_iou, best_box = iou, det
        results.append((frame, vis_base, dets, gt_box, best_iou, best_box, h_orig, w_orig))

    if not results:
        return 0

    # Determine fallback and zero-case
    max_iou = max(r[4] for r in results)
    all_zero = all(r[4] == 0 for r in results)
    fallback = max_iou < FALLBACK_THRESHOLD and not all_zero

    # Print object header
    print(f"\n{Colors.BOLD}{Colors.CYAN}📦 Object: {Colors.RESET}{Colors.BOLD}{name}{Colors.RESET}")
    
    updated = {} if fallback else dict(gt_data)
    updated_count = 0
    out_img_dir = BBOX_CACHE_DIR / name / "images"
    out_img_dir.mkdir(parents=True, exist_ok=True)

    # Log and save results - show ALL IOU values
    with open(log_file, 'a') as lf:
        for frame, vis0, dets, gt_box, best_iou, best_box, h_orig, w_orig in results:
            if all_zero:
                keep = False
            elif fallback:
                keep = bool(best_box and best_iou > MIN_IOU_SAVE)
            else:
                keep = bool(best_box and best_iou >= IOU_THRESHOLD)

            lf.write(f"{frame},{best_iou:.3f},{keep}\n")
            
            # Show all IOU values with color coding
            if keep:
                status_icon = f"{Colors.GREEN}✓{Colors.RESET}"
                status_text = f"{Colors.GREEN}updated{Colors.RESET}"
            else:
                status_icon = f"{Colors.YELLOW}○{Colors.RESET}"
                status_text = f"{Colors.YELLOW}kept original{Colors.RESET}"
            
            # Format IOU value with color based on threshold
            if best_iou >= IOU_THRESHOLD:
                iou_color = Colors.GREEN
            elif best_iou >= FALLBACK_THRESHOLD:
                iou_color = Colors.YELLOW
            else:
                iou_color = Colors.RED
            
            print(f"  {status_icon} {Colors.BOLD}{frame}{Colors.RESET} → IOU: {iou_color}{best_iou:.3f}{Colors.RESET} ({status_text})")

            if keep:
                # Convert back to original orientation before saving
                unrotated_box = rotate_box_ccw_xyxy(best_box, h_orig, w_orig)
                updated[frame] = unrotated_box
                draw_box = best_box
                updated_count += 1
            else:
                if fallback:
                    continue
                # For visualization, we use rotated boxes
                updated[frame] = gt_data[frame]  # Keep original box
                draw_box = gt_box

            # Draw and save image
            vis = vis0.copy()
            for det in dets:
                cv2.rectangle(vis, tuple(det[:2]), tuple(det[2:]), (0, 255, 0), 2)
            cv2.rectangle(vis, tuple(draw_box[:2]), tuple(draw_box[2:]), (0, 0, 255), 3)

            # Reduce size to half height and width to save space in cache
            h, w = vis.shape[:2]
            vis = cv2.resize(vis, (w // 2, h // 2), interpolation=cv2.INTER_AREA)

            # Save as JPG (smaller file size for streaming)
            cv2.imwrite(str(out_img_dir / f"{frame}.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 85])

    # Save updated JSON
    out_json = folder / f"{gt_json.stem}_updated.json"
    out_json.write_text(json.dumps(updated, indent=2))
    
    # Summary line without warnings
    total_frames = len(results)
    print(f"  {Colors.CYAN}Summary:{Colors.RESET} {Colors.BOLD}{updated_count}/{total_frames}{Colors.RESET} frames updated → {Colors.CYAN}{out_json}{Colors.RESET}")

    return updated_count


def main():
    import time
    start_time = time.time()
    
    args = parse_args()
    print(f"\n{Colors.BOLD}{Colors.MAGENTA}╔═══════════════════════════════════════════════════════════════╗{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.MAGENTA}║{Colors.RESET} {Colors.BOLD}{Colors.CYAN}🎯 Bounding Box Polish Pipeline{Colors.RESET} {Colors.BOLD}{Colors.MAGENTA}║{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.MAGENTA}╚═══════════════════════════════════════════════════════════════╝{Colors.RESET}")
    print(f"{Colors.CYAN}Processing scene:{Colors.RESET} {Colors.BOLD}{args.scene}{Colors.RESET}")
    print(f"{Colors.CYAN}Input directory:{Colors.RESET} {Colors.CYAN}{INPUT_DIR}{Colors.RESET}\n")
    
    # Check if model weights file exists
    if not os.path.exists(MODEL_PATH):
        print(f"\n{Colors.RED}❌ Error:{Colors.RESET} Model weights file not found at: {Colors.CYAN}{MODEL_PATH}{Colors.RESET}")
        print(f"\n{Colors.YELLOW}Please download the GroundingDINO weights:{Colors.RESET}")
        print(f"  {Colors.CYAN}mkdir -p {os.path.dirname(MODEL_PATH)}{Colors.RESET}")
        print(f"  {Colors.CYAN}wget -O {MODEL_PATH} https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth{Colors.RESET}")
        print(f"\nOr if the weights are in a different location, update MODEL_PATH in the script.")
        return
    
    # Check if config file exists
    if not os.path.exists(CONFIG_PATH):
        print(f"\n{Colors.RED}❌ Error:{Colors.RESET} Config file not found at: {Colors.CYAN}{CONFIG_PATH}{Colors.RESET}")
        return
    
    print(f"{Colors.BLUE}🔄{Colors.RESET} Loading GroundingDINO model...")
    model = load_model(CONFIG_PATH, MODEL_PATH)
    print(f"{Colors.GREEN}✓{Colors.RESET} Model loaded successfully\n")
    
    total = 0
    
    # Check if the input directory exists
    if not INPUT_DIR.exists():
        print(f"{Colors.RED}❌ Error:{Colors.RESET} Input directory '{Colors.CYAN}{INPUT_DIR}{Colors.RESET}' does not exist.")
        print("Please provide a valid scene name.")
        return
    
    # Process all folders
    folders = [subdir for subdir in sorted(INPUT_DIR.iterdir()) if subdir.is_dir()]
    print(f"{Colors.CYAN}Found {Colors.BOLD}{len(folders)}{Colors.RESET}{Colors.CYAN} objects to process{Colors.RESET}\n")
    
    for subdir in folders:
        total += process_folder(subdir, model)
    
    elapsed_time = time.time() - start_time
    print(f"\n{Colors.BOLD}{Colors.GREEN}╔═══════════════════════════════════════════════════════════════╗{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GREEN}║{Colors.RESET} {Colors.BOLD}✓ Bounding box polish completed!{Colors.RESET} {Colors.BOLD}{Colors.GREEN}║{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GREEN}╚═══════════════════════════════════════════════════════════════╝{Colors.RESET}")
    print(f"{Colors.CYAN}Total updated boxes:{Colors.RESET} {Colors.BOLD}{total}{Colors.RESET}")
    print(f"{Colors.CYAN}Total time:{Colors.RESET} {Colors.BOLD}{elapsed_time:.2f}s{Colors.RESET}\n")


if __name__ == "__main__":
    main()
