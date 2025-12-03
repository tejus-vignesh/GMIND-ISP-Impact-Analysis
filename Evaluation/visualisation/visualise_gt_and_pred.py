#!/usr/bin/env python3
"""
Visualise ground truth and predictions on the dataset in real-time.

Displays ground truth and prediction annotations side-by-side or overlaid
on video frames. Useful for debugging detection issues and understanding
model performance visually.

Features:
    - Side-by-side or overlaid GT and prediction visualisation
    - Highlights missed large GT objects in orange
    - Real-time playback with keyboard controls
    - Frame-by-frame navigation

Controls:
    - Space: Pause/Resume playback
    - Q or ESC: Quit
    - Right Arrow / D: Step forward one frame (when paused)
    - Left Arrow / A: Step backward one frame (when paused)

Usage:
    python -m Evaluation.visualisation.visualise_gt_and_pred \\
        --gt-file baseline_evaluation_results/coco_gt.json \\
        --results baseline_evaluation_results/coco_results.json \\
        --output-dir baseline_evaluation_results

Note:
    Requires access to the original video files from the dataset.
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from pycocotools.coco import COCO

from Evaluation.core.baseline_detector_and_tracker import get_gmind_dataloader, load_config


def compute_iou(box1, box2):
    """Compute IoU between two boxes in [x, y, w, h] format"""
    x1_1, y1_1, w1, h1 = box1
    x2_1, y2_1 = x1_1 + w1, y1_1 + h1

    x1_2, y1_2, w2, h2 = box2
    x2_2, y2_2 = x1_2 + w2, y1_2 + h2

    # Intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0

    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    return intersection / union


def get_area_category(area):
    """Get COCO area category"""
    if area < 32**2:
        return "small"
    elif area < 96**2:
        return "medium"
    else:
        return "large"


def visualise_frame(image_np, gt_boxes, pred_boxes, category_id_to_name, max_height=1080):
    """
    Visualise GT and predictions on a frame.
    Highlights missed large GT objects in orange.

    Args:
        image_np: Image as numpy array (H, W, C) in RGB format
        gt_boxes: List of GT boxes with format [x1, y1, x2, y2, category_id, area] (area optional)
        pred_boxes: List of prediction boxes with format [x1, y1, x2, y2, category_id, score]
        category_id_to_name: Mapping from category_id to name
        max_height: Maximum height for display

    Returns:
        Image with visualizations as numpy array (BGR format for OpenCV)
    """
    orig_h, orig_w = image_np.shape[:2]

    # Downsample if image is larger than max_height
    scale_factor = 1.0
    if orig_h > max_height:
        scale_factor = max_height / orig_h
        new_h = max_height
        new_w = int(orig_w * scale_factor)
        image_np = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Convert RGB to BGR for OpenCV
    vis_image = cv2.cvtColor(image_np.copy(), cv2.COLOR_RGB2BGR)

    # Match predictions to GT to identify missed large objects
    # Convert boxes to [x, y, w, h] format for IoU computation
    matched_gt_indices = set()

    for pred_idx, pred_box in enumerate(pred_boxes):
        px1, py1, px2, py2, pcat_id, pscore = pred_box
        pred_bbox = [px1, py1, px2 - px1, py2 - py1]  # [x, y, w, h]

        best_iou = 0.0
        best_gt_idx = None

        for gt_idx, gt_box in enumerate(gt_boxes):
            if len(gt_box) >= 5:
                gx1, gy1, gx2, gy2, gcat_id = gt_box[:5]
            else:
                continue

            if gcat_id != pcat_id:
                continue

            gt_bbox = [gx1, gy1, gx2 - gx1, gy2 - gy1]  # [x, y, w, h]
            iou = compute_iou(pred_bbox, gt_bbox)

            if iou > best_iou and iou >= 0.5:  # Match threshold
                best_iou = iou
                best_gt_idx = gt_idx

        if best_gt_idx is not None:
            matched_gt_indices.add(best_gt_idx)

    # Draw ground truth boxes
    # Green (dashed) for matched GT, Orange (dashed) for missed large GT
    for gt_idx, box in enumerate(gt_boxes):
        if len(box) >= 5:
            x1, y1, x2, y2, cat_id = box[:5]
            area = box[5] if len(box) > 5 else (x2 - x1) * (y2 - y1)
        else:
            x1, y1, x2, y2, cat_id = box
            area = (x2 - x1) * (y2 - y1)

        x1, y1, x2, y2 = (
            int(x1 * scale_factor),
            int(y1 * scale_factor),
            int(x2 * scale_factor),
            int(y2 * scale_factor),
        )

        # Clamp to image bounds
        x1 = max(0, min(x1, vis_image.shape[1] - 1))
        y1 = max(0, min(y1, vis_image.shape[0] - 1))
        x2 = max(0, min(x2, vis_image.shape[1] - 1))
        y2 = max(0, min(y2, vis_image.shape[0] - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        # Determine color: orange for missed large objects, green for matched
        is_matched = gt_idx in matched_gt_indices
        is_large = get_area_category(area) == "large"
        is_missed_large = is_large and not is_matched

        if is_missed_large:
            color = (0, 165, 255)  # Orange in BGR
            label_prefix = "MISSED: "
        else:
            color = (0, 255, 0)  # Green in BGR
            label_prefix = "GT: "

        # Draw dashed box
        dash_length = 10
        gap_length = 5
        # Top and bottom
        for x in range(x1, x2, dash_length + gap_length):
            cv2.line(vis_image, (x, y1), (min(x + dash_length, x2), y1), color, 2)
            cv2.line(vis_image, (x, y2), (min(x + dash_length, x2), y2), color, 2)
        # Left and right
        for y in range(y1, y2, dash_length + gap_length):
            cv2.line(vis_image, (x1, y), (x1, min(y + dash_length, y2)), color, 2)
            cv2.line(vis_image, (x2, y), (x2, min(y + dash_length, y2)), color, 2)

        # Add label
        cat_name = category_id_to_name.get(cat_id, f"Class {cat_id}")
        label_text = f"{label_prefix}{cat_name}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        # Draw text background
        cv2.rectangle(
            vis_image, (x1, y1 - text_height - 5), (x1 + text_width, y1), (255, 255, 255), -1
        )
        cv2.putText(vis_image, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Draw predictions in red (solid)
    for box in pred_boxes:
        x1, y1, x2, y2, cat_id, score = box
        x1, y1, x2, y2 = (
            int(x1 * scale_factor),
            int(y1 * scale_factor),
            int(x2 * scale_factor),
            int(y2 * scale_factor),
        )

        # Clamp to image bounds
        x1 = max(0, min(x1, vis_image.shape[1] - 1))
        y1 = max(0, min(y1, vis_image.shape[0] - 1))
        x2 = max(0, min(x2, vis_image.shape[1] - 1))
        y2 = max(0, min(y2, vis_image.shape[0] - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        # Draw solid box
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Add label with confidence
        cat_name = category_id_to_name.get(cat_id, f"Class {cat_id}")
        label_text = f"Pred: {cat_name} {score:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        # Draw text background
        cv2.rectangle(
            vis_image, (x1, y2), (x1 + text_width, y2 + text_height + 5), (255, 255, 255), -1
        )
        cv2.putText(
            vis_image,
            label_text,
            (x1, y2 + text_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )

    # Add title and frame info
    title_text = (
        "Green (dashed): Matched GT | Orange (dashed): Missed Large GT | Red (solid): Predictions"
    )
    cv2.putText(vis_image, title_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return vis_image


def main():
    parser = argparse.ArgumentParser(description="Visualise GT and predictions in real-time")
    parser.add_argument("--results", type=str, required=True, help="Path to COCO results JSON file")
    parser.add_argument("--gt-file", type=str, required=True, help="Path to COCO GT JSON file")
    parser.add_argument(
        "--output-dir", type=str, default="baseline_evaluation_results", help="Output directory"
    )
    parser.add_argument("--fps", type=float, default=10.0, help="Display FPS (default: 10.0)")
    parser.add_argument("--start-frame", type=int, default=0, help="Start frame index (default: 0)")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to display (default: all)",
    )
    args = parser.parse_args()

    # Load GT
    gt_path = Path(args.gt_file)
    if not gt_path.exists():
        gt_path = Path(args.output_dir) / gt_path.name

    print(f"Loading GT from: {gt_path}")
    coco_gt = COCO(str(gt_path))

    # Get category mapping
    categories = coco_gt.loadCats(coco_gt.getCatIds())
    category_id_to_name = {cat["id"]: cat["name"] for cat in categories}
    category_name_to_id = {cat["name"]: cat["id"] for cat in categories}

    # Load results
    results_path = Path(args.results)
    if not results_path.exists():
        results_path = Path(args.output_dir) / results_path.name

    print(f"Loading results from: {results_path}")
    with open(results_path, "r") as f:
        coco_results = json.load(f)

    # Group results by image_id
    results_by_image = defaultdict(list)
    for result in coco_results:
        results_by_image[result["image_id"]].append(result)

    # Load dataset
    print("\nLoading dataset...")
    config_path = Path(__file__).parent / "DeepLearning" / "gmind_config.yaml"
    config = load_config(config_path)

    data_root = Path(config["data"]["root"])
    sensor = config["data"].get("sensor", "FLIR8.9")
    frame_stride = config["data"].get("frame_stride", 1)
    max_frames_per_video = config["data"].get("max_frames_per_video")

    test_config = config.get("test", {})
    test_sets = test_config.get("sets", [])
    test_set_subdirs = test_config.get("set_subdirs", {})

    test_loader = get_gmind_dataloader(
        data_root=data_root,
        sets=test_sets,
        sensor=sensor,
        transforms=None,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        frame_stride=frame_stride,
        max_frames=max_frames_per_video,
        set_subdirs=test_set_subdirs,
    )
    test_dataset = test_loader.dataset

    print(f"Dataset loaded: {len(test_dataset)} frames")
    print(f"\nControls:")
    print(f"  'q' - Quit")
    print(f"  'p' - Pause/Resume")
    print(f"  'n' - Next frame")
    print(f"  'b' - Previous frame")
    print(f"  SPACE - Play/Pause")
    print(f"  '+' - Increase speed")
    print(f"  '-' - Decrease speed")
    print(f"\nStarting visualization...")

    # Visualization state
    paused = False
    frame_delay = 1.0 / args.fps
    current_frame = args.start_frame
    max_frame = len(test_dataset) - 1
    if args.max_frames:
        max_frame = min(max_frame, args.start_frame + args.max_frames - 1)

    fps_start_time = time.time()
    fps_frame_count = 0
    fps = 0.0

    try:
        for batch_idx in range(args.start_frame, max_frame + 1):
            if batch_idx >= len(test_dataset):
                break

            # Get frame
            images, targets = test_dataset[batch_idx]
            image = images[0] if isinstance(images, (list, tuple)) else images
            target = targets[0] if isinstance(targets, (list, tuple)) else targets

            # Convert to numpy
            if isinstance(image, torch.Tensor):
                if image.dim() == 4:
                    image = image[0]
                image_np = image.permute(1, 2, 0).cpu().numpy()
                if image_np.max() <= 1.0:
                    image_np = (image_np * 255).astype(np.uint8)
                else:
                    image_np = image_np.astype(np.uint8)
            else:
                image_np = np.array(image)

            # Get image_id (frame_idx + 1)
            image_id = batch_idx + 1

            # Get GT boxes for this image
            gt_ann_ids = coco_gt.getAnnIds(imgIds=[image_id])
            gt_anns = coco_gt.loadAnns(gt_ann_ids)
            gt_boxes = []
            for ann in gt_anns:
                bbox = ann["bbox"]  # [x, y, w, h]
                x1, y1, w, h = bbox
                x2, y2 = x1 + w, y1 + h
                area = ann.get("area", w * h)
                gt_boxes.append([x1, y1, x2, y2, ann["category_id"], area])

            # Get prediction boxes for this image
            pred_boxes = []
            if image_id in results_by_image:
                for result in results_by_image[image_id]:
                    bbox = result["bbox"]  # [x, y, w, h]
                    x1, y1, w, h = bbox
                    x2, y2 = x1 + w, y1 + h
                    pred_boxes.append([x1, y1, x2, y2, result["category_id"], result["score"]])

            # Create visualization
            vis_image = visualise_frame(image_np, gt_boxes, pred_boxes, category_id_to_name)

            # Add frame info
            frame_text = f"Frame {batch_idx + 1}/{len(test_dataset)} | GT: {len(gt_boxes)} | Pred: {len(pred_boxes)} | FPS: {fps:.1f}"
            cv2.putText(
                vis_image,
                frame_text,
                (10, vis_image.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            # Display
            cv2.imshow("GT and Predictions", vis_image)

            # Calculate FPS
            fps_frame_count += 1
            if fps_frame_count % 10 == 0:
                elapsed = time.time() - fps_start_time
                fps = 10.0 / elapsed if elapsed > 0 else 0.0
                fps_start_time = time.time()

            # Handle keyboard input
            if paused:
                key = cv2.waitKey(0) & 0xFF
            else:
                key = cv2.waitKey(int(frame_delay * 1000)) & 0xFF

            if key == ord("q"):
                print("\nQuitting...")
                break
            elif key == ord("p") or key == ord(" "):
                paused = not paused
                print(f"{'Paused' if paused else 'Resumed'}")
            elif key == ord("n"):
                # Next frame
                continue
            elif key == ord("b"):
                # Previous frame
                batch_idx = max(args.start_frame, batch_idx - 2)  # -2 because loop will increment
                continue
            elif key == ord("+") or key == ord("="):
                frame_delay = max(0.01, frame_delay * 0.9)
                args.fps = 1.0 / frame_delay
                print(f"Speed increased: {args.fps:.1f} FPS")
            elif key == ord("-") or key == ord("_"):
                frame_delay = min(1.0, frame_delay * 1.1)
                args.fps = 1.0 / frame_delay
                print(f"Speed decreased: {args.fps:.1f} FPS")

            if not paused:
                time.sleep(frame_delay)

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        cv2.destroyAllWindows()
        print("Visualization closed")


if __name__ == "__main__":
    main()
