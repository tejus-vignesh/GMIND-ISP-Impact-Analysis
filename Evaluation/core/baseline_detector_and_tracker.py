#!/usr/bin/env python3
"""
Baseline Detector and Tracker Evaluation Script.

This is the main evaluation script for the GMIND dataset. It uses YOLOv11n
for object detection and OC-SORT for multi-object tracking, then computes
comprehensive evaluation metrics following COCO evaluation protocols.

Features:
    - Object detection using YOLOv11n (Ultralytics)
    - Multi-object tracking using OC-SORT
    - COCO format evaluation metrics (AP50, AP50-95, APsmall, APlarge, etc.)
    - Per-class and per-video metric breakdowns
    - Real-time visualisation of detections and tracking
    - Checkpointing for resuming interrupted evaluations

Metrics Computed:
    - AP50-95: Average Precision over IoU thresholds 0.5:0.05:0.95
    - AP50: Average Precision at IoU threshold 0.5
    - AP75: Average Precision at IoU threshold 0.75
    - APsmall/APmedium/APlarge: AP for different object sizes
    - Per-class AP50: AP50 for each object class separately
    - Per-video metrics: All above metrics computed per video

Usage:
    python -m Evaluation.core.baseline_detector_and_tracker \\
        --config DeepLearning/gmind_config.yaml \\
        --output-dir evaluation_results

Requirements:
    - YOLOv11n model (automatically downloaded if not present)
    - OC-SORT tracker (must be installed separately)
    - pycocotools for COCO evaluation
    - GMIND dataset configured in config file
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
import numpy as np
import torch
import yaml
from tqdm import tqdm

# Try to use an interactive backend if display is available
# Only use Agg (non-interactive) if no display is available or backends fail
has_display = os.environ.get("DISPLAY") is not None and os.environ.get("DISPLAY")
backend_set = False

if has_display:
    # Try interactive backends in order of preference
    for backend in ["TkAgg", "Qt5Agg", "Qt4Agg"]:
        try:
            matplotlib.use(backend, force=False)
            backend_set = True
            break
        except (ImportError, ModuleNotFoundError, Exception):
            # Backend not available (e.g., tkinter missing), try next one
            continue

    if not backend_set:
        # All interactive backends failed, use Agg
        matplotlib.use("Agg")
else:
    # No display available, use non-interactive backend
    matplotlib.use("Agg")

# OC-SORT imports (required)
import sys
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt

# YOLO imports
from ultralytics import YOLO

# Try to find OC-SORT in common locations
# OC_SORT is a git submodule at the repository root
repo_root = Path(__file__).parent.parent.parent  # Go from Evaluation/core/ to repo root
ocsort_paths = [
    repo_root / "OC_SORT",  # Root-level submodule (preferred)
    Path("/tmp/OC_SORT"),
    Path(__file__).parent / "OC_SORT",  # Local to this script
    Path.home() / "OC_SORT",
]

OC_SORT_PATH = None
for path in ocsort_paths:
    if path.exists() and (path / "trackers" / "ocsort_tracker" / "ocsort.py").exists():
        OC_SORT_PATH = path
        sys.path.insert(0, str(path))
        break

try:
    # Try different possible import paths for OC-SORT
    try:
        from ocsort import OCSort
    except ImportError:
        try:
            from ocsort.ocsort import OCSort
        except ImportError:
            try:
                from trackers.ocsort_tracker.ocsort import OCSort
            except ImportError:
                from tracker.ocsort import OCSort
except ImportError:
    raise ImportError(
        "OC-SORT is required but not installed. "
        "Install from: https://github.com/noahcao/OC_SORT\n"
        "Clone the repository and add it to your Python path, or install via pip if available."
    )

# GMIND DataLoader
from DataLoader import get_gmind_dataloader

# COCO evaluation
try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    COCO_EVAL_AVAILABLE = True
except ImportError:
    COCO_EVAL_AVAILABLE = False
    print("Warning: pycocotools not available. Install with: pip install pycocotools")


def load_config(config_path: Path) -> Dict:
    """Load GMIND config YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def create_coco_gt_from_dataset(dataset) -> Tuple[COCO, str, Dict[str, int], Dict[int, int]]:
    """
    Create COCO ground truth object from GMIND dataset.

    Args:
        dataset: GMINDDataset instance

    Returns:
        Tuple of (COCO object with ground truth annotations, path to temporary JSON file, category_name_to_id mapping)
    """
    if not COCO_EVAL_AVAILABLE:
        raise RuntimeError("pycocotools is required for COCO evaluation")

    # Build COCO format data
    coco_data = {
        "info": {"description": "GMIND Dataset", "version": "1.0", "year": 2024},
        "images": [],
        "annotations": [],
        "categories": [],
    }

    # Get categories from all annotation files to ensure we have all categories
    all_categories = {}  # cat_id -> cat dict
    category_name_to_id = {}  # cat_name (lowercase) -> cat_id

    for video_item in dataset.video_items:
        ann_path = video_item["annotation_path"]
        with open(ann_path, "r") as f:
            ann_data = json.load(f)
            for cat in ann_data.get("categories", []):
                cat_id = cat["id"]
                cat_name = cat["name"].lower()  # Use lowercase for matching
                if cat_id not in all_categories:
                    all_categories[cat_id] = cat
                # Store name mapping
                if cat_name not in category_name_to_id:
                    category_name_to_id[cat_name] = cat_id

    # Convert to list format for COCO
    coco_data["categories"] = [cat for cat in all_categories.values()]

    # Build image and annotation lists
    # Use frame_idx + 1 as unique image_id to avoid collisions between videos
    # Store mapping from original image_id to unique image_id per video
    ann_id = 1
    frame_idx_to_unique_image_id = {}  # frame_idx -> unique_image_id

    for frame_idx, frame_info in enumerate(dataset.frame_index):
        video_item = dataset.video_items[frame_info["video_idx"]]
        image_info = frame_info.get("image_info")

        # Always use frame_idx + 1 as unique image_id to avoid collisions
        unique_image_id = frame_idx + 1
        frame_idx_to_unique_image_id[frame_idx] = unique_image_id

        if image_info is None:
            # Get image dimensions from video
            cap = cv2.VideoCapture(str(video_item["video_path"]))
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
            else:
                width, height = 1920, 1080  # Default
        else:
            width = image_info["width"]
            height = image_info["height"]

        # Add image with unique image_id
        coco_data["images"].append(
            {
                "id": unique_image_id,
                "width": width,
                "height": height,
                "file_name": f"frame_{frame_idx:06d}.jpg",
            }
        )

        # Add annotations with unique image_id
        for ann in frame_info["annotations"]:
            bbox = ann["bbox"]  # [x, y, width, height]
            coco_data["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": unique_image_id,
                    "category_id": ann["category_id"],
                    "bbox": bbox,
                    "area": ann.get("area", bbox[2] * bbox[3]),
                    "iscrowd": ann.get("iscrowd", 0),
                }
            )
            ann_id += 1

    # Create temporary JSON file for COCO
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(coco_data, f)
        temp_gt_path = f.name

    # Load as COCO object
    coco_gt = COCO(temp_gt_path)

    # category_name_to_id was already created above
    # frame_idx_to_unique_image_id was created above
    return coco_gt, temp_gt_path, category_name_to_id, frame_idx_to_unique_image_id


def convert_yolo_to_coco_format(
    yolo_results, image_ids: List[int], model, category_name_to_id: Dict[str, int]
) -> List[Dict]:
    """
    Convert YOLO detection results to COCO format.

    Args:
        yolo_results: List of YOLO result objects
        image_ids: List of image IDs corresponding to each result
        model: YOLO model (to get class names)
        category_name_to_id: Mapping from category name (lowercase) to category ID

    Returns:
        List of COCO-format detection dictionaries
    """
    coco_results = []

    # Get YOLO class names
    yolo_class_names = model.names if hasattr(model, "names") else {}

    # Create mapping from YOLO class ID to dataset category ID
    # Only map the classes we actually use: person (0), bicycle (1), car (2)
    yolo_cls_to_category_id = {}
    classes_to_map = [0, 1, 2]  # person, bicycle, car

    for yolo_cls_id in classes_to_map:
        if yolo_cls_id in yolo_class_names:
            yolo_cls_name = yolo_class_names[yolo_cls_id]
            yolo_cls_name_lower = yolo_cls_name.lower()
            if yolo_cls_name_lower in category_name_to_id:
                yolo_cls_to_category_id[yolo_cls_id] = category_name_to_id[yolo_cls_name_lower]
            else:
                # Fallback: try adding 1 (standard COCO mapping)
                yolo_cls_to_category_id[yolo_cls_id] = int(yolo_cls_id + 1)
                print(
                    f"Warning: Category '{yolo_cls_name}' not found in dataset categories. Using fallback ID {yolo_cls_to_category_id[yolo_cls_id]}"
                )

    for result, image_id in zip(yolo_results, image_ids):
        # Check if we have filtered data (from class filtering)
        if hasattr(result, "_filtered_data") and result._filtered_data is not None:
            # Use filtered data
            boxes = result._filtered_data["boxes"]
            scores = result._filtered_data["scores"]
            classes = result._filtered_data["classes"].astype(int)
        elif result.boxes is not None and len(result.boxes) > 0:
            # Use original boxes (no filtering applied)
            boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
        else:
            # No detections
            continue

        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1

            # Map YOLO class ID to dataset category ID
            cls_int = int(cls)
            if cls_int in yolo_cls_to_category_id:
                category_id = yolo_cls_to_category_id[cls_int]
            else:
                # Fallback: add 1
                category_id = int(cls + 1)
                print(
                    f"Warning: YOLO class {cls_int} not in mapping. Using fallback ID {category_id}"
                )

            coco_results.append(
                {
                    "image_id": int(image_id),
                    "category_id": category_id,
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "score": float(score),
                }
            )

    return coco_results


def compute_coco_metrics(
    coco_gt: COCO, coco_results: List[Dict]
) -> Tuple[Dict, Optional[COCOeval]]:
    """
    Compute COCO evaluation metrics.

    Returns:
        Tuple of (Dictionary with metrics: AP50, AP50-95, APsmall, APlarge, etc., COCOeval object or None)
    """
    if not COCO_EVAL_AVAILABLE:
        raise RuntimeError("pycocotools is required for COCO evaluation")

    # Validate results before loading
    if len(coco_results) == 0:
        print("Warning: No detections to evaluate")
        return {
            "AP50-95": 0.0,
            "AP50": 0.0,
            "AP75": 0.0,
            "APsmall": 0.0,
            "APmedium": 0.0,
            "APlarge": 0.0,
            "AR50-95": 0.0,
        }, None

    # Check for category ID mismatches
    gt_cat_ids = set(coco_gt.getCatIds())
    pred_cat_ids = set(r["category_id"] for r in coco_results)
    invalid_cat_ids = pred_cat_ids - gt_cat_ids

    if invalid_cat_ids:
        print(
            f"Warning: Found {len(invalid_cat_ids)} invalid category IDs in predictions: {invalid_cat_ids}"
        )
        print(f"Valid category IDs in GT: {sorted(gt_cat_ids)}")
        # Filter out invalid category IDs
        coco_results = [r for r in coco_results if r["category_id"] in gt_cat_ids]
        print(f"Filtered to {len(coco_results)} valid detections")

    if len(coco_results) == 0:
        print("Warning: No valid detections after filtering")
        return {
            "AP50-95": 0.0,
            "AP50": 0.0,
            "AP75": 0.0,
            "APsmall": 0.0,
            "APmedium": 0.0,
            "APlarge": 0.0,
            "AR50-95": 0.0,
        }, None

    # Check for image ID mismatches
    gt_img_ids = set(coco_gt.getImgIds())
    pred_img_ids = set(r["image_id"] for r in coco_results)
    invalid_img_ids = pred_img_ids - gt_img_ids

    if invalid_img_ids:
        print(f"Warning: Found {len(invalid_img_ids)} invalid image IDs in predictions")
        # Filter out invalid image IDs
        coco_results = [r for r in coco_results if r["image_id"] in gt_img_ids]
        print(f"Filtered to {len(coco_results)} detections with valid image IDs")

    try:
        # Load detections
        coco_dt = coco_gt.loadRes(coco_results)

        # Run evaluation
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    except Exception as e:
        print(f"Error during COCO evaluation: {e}")
        print(f"GT has {len(coco_gt.imgs)} images, {len(coco_gt.anns)} annotations")
        print(f"Predictions have {len(coco_results)} detections")
        raise

    # Extract metrics
    # COCOeval.stats format:
    # [0] AP @ IoU=0.50:0.95 (all categories, all areas)
    # [1] AP @ IoU=0.50 (all categories, all areas)
    # [2] AP @ IoU=0.75 (all categories, all areas)
    # [3] AP @ IoU=0.50:0.95 (small objects)
    # [4] AP @ IoU=0.50:0.95 (medium objects)
    # [5] AP @ IoU=0.50:0.95 (large objects)
    # [6] AR @ IoU=0.50:0.95 (all categories, all areas, maxDets=1)
    # [7] AR @ IoU=0.50:0.95 (all categories, all areas, maxDets=10)
    # [8] AR @ IoU=0.50:0.95 (all categories, all areas, maxDets=100)
    # [9] AR @ IoU=0.50:0.95 (small objects)
    # [10] AR @ IoU=0.50:0.95 (medium objects)
    # [11] AR @ IoU=0.50:0.95 (large objects)

    metrics = {
        "AP50-95": coco_eval.stats[0],
        "AP50": coco_eval.stats[1],
        "AP75": coco_eval.stats[2],
        "APsmall": coco_eval.stats[3],
        "APmedium": coco_eval.stats[4],
        "APlarge": coco_eval.stats[5],
        "AR50-95": coco_eval.stats[6],
    }

    return metrics, coco_eval


def compute_per_class_ap(coco_gt: COCO, coco_dt: COCO) -> Dict[str, float]:
    """Compute AP per class."""
    per_class_ap = {}

    # Get category names
    cat_ids = coco_gt.getCatIds()
    cat_info = coco_gt.loadCats(cat_ids)
    cat_names = {cat["id"]: cat["name"] for cat in cat_info}

    # Evaluate per category - create fresh COCOeval for each
    for cat_id in cat_ids:
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
        coco_eval.params.catIds = [cat_id]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # AP50-95 for this category
        ap = coco_eval.stats[0]
        cat_name = cat_names[cat_id]
        per_class_ap[cat_name] = ap

    return per_class_ap


def compute_per_video_metrics(
    dataset,
    coco_results: List[Dict],
    coco_gt: COCO,
    frame_idx_to_unique_image_id: Optional[Dict[int, int]] = None,
) -> Dict[str, Dict]:
    """
    Compute AP metrics per video.

    Args:
        dataset: GMINDDataset instance
        coco_results: List of COCO-format detection results
        coco_gt: COCO ground truth object
        frame_idx_to_unique_image_id: Optional mapping from frame_idx to unique image_id
                                     If None, assumes image_ids are already unique (frame_idx + 1)

    Returns:
        Dictionary mapping video name to metrics
    """
    per_video_metrics = {}

    # Check if results use unique image IDs or overlapping ones
    # If frame_idx_to_unique_image_id is provided, results should use unique IDs
    # Otherwise, we need to handle overlapping IDs differently

    # Group results by video using frame indices
    video_frame_map = {}  # video_idx -> list of (frame_idx, image_id_in_gt)

    for frame_idx, frame_info in enumerate(dataset.frame_index):
        video_idx = frame_info["video_idx"]

        # Determine the image_id used in COCO GT for this frame
        if frame_idx_to_unique_image_id is not None:
            # Results use unique image IDs (frame_idx + 1)
            image_id_in_gt = frame_idx_to_unique_image_id.get(frame_idx, frame_idx + 1)
        else:
            # Results might use original overlapping IDs - use frame_idx + 1 as unique ID
            image_id_in_gt = frame_idx + 1

        if video_idx not in video_frame_map:
            video_frame_map[video_idx] = []
        video_frame_map[video_idx].append((frame_idx, image_id_in_gt))

    # Evaluate each video separately
    for video_idx, frame_list in video_frame_map.items():
        video_item = dataset.video_items[video_idx]
        video_name = (
            f"{video_item['set_name']}/{video_item['subdir']}/{video_item['video_path'].name}"
        )

        # Get image IDs for this video (unique IDs in COCO GT)
        video_image_ids = set([img_id for _, img_id in frame_list])

        # Filter results for this video
        video_results = [r for r in coco_results if r["image_id"] in video_image_ids]

        if len(video_results) == 0:
            print(f"Warning: No results found for video {video_name}")
            continue

        print(f"Video {video_name}: {len(video_image_ids)} images, {len(video_results)} detections")

        # Create temporary COCO objects for this video
        # Get ground truth images and annotations for this video
        video_gt_images = coco_gt.loadImgs(video_image_ids)
        video_gt_anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=video_image_ids))

        # Create video-specific COCO GT
        video_coco_data = {
            "info": {"description": "GMIND Dataset - Video Subset", "version": "1.0", "year": 2024},
            "images": video_gt_images,
            "annotations": video_gt_anns,
            "categories": coco_gt.loadCats(coco_gt.getCatIds()),
        }

        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(video_coco_data, f)
            temp_video_gt_path = f.name

        video_coco_gt = COCO(temp_video_gt_path)
        video_coco_dt = video_coco_gt.loadRes(video_results)

        # Evaluate
        video_coco_eval = COCOeval(video_coco_gt, video_coco_dt, iouType="bbox")
        video_coco_eval.evaluate()
        video_coco_eval.accumulate()
        video_coco_eval.summarize()

        per_video_metrics[video_name] = {
            "AP50-95": video_coco_eval.stats[0],
            "AP50": video_coco_eval.stats[1],
            "APsmall": video_coco_eval.stats[3],
            "APlarge": video_coco_eval.stats[5],
        }

    return per_video_metrics


def visualise_realtime(
    image_np, tracked_detections, model_names=None, frame_num=0, fps=0.0, max_height=1080
):
    """
    Visualise real-time detections with tracking IDs.

    Args:
        image_np: Image as numpy array (H, W, C) in RGB format
        tracked_detections: Array from OC-SORT tracker [x1, y1, x2, y2, track_id, class_id, score]
        model_names: Optional class name mapping (from YOLO model)
        frame_num: Current frame number
        fps: Current FPS
        max_height: Maximum height for display (default: 1080p)

    Returns:
        Image with visualisations as numpy array (BGR format for OpenCV)
    """
    import time

    # Downsample if image is larger than max_height
    orig_h, orig_w = image_np.shape[:2]
    scale_factor = 1.0

    if orig_h > max_height:
        scale_factor = max_height / orig_h
        new_h = max_height
        new_w = int(orig_w * scale_factor)
        image_np = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Convert RGB to BGR for OpenCV
    vis_image = cv2.cvtColor(image_np.copy(), cv2.COLOR_RGB2BGR)

    # Colour palette for different classes (BGR format)
    class_colours = {
        0: (0, 0, 255),  # person - red
        1: (255, 0, 0),  # bicycle - blue
        2: (0, 255, 0),  # car - green
    }

    # Draw tracked detections
    num_detections = 0
    if tracked_detections is not None and len(tracked_detections) > 0:
        for det in tracked_detections:
            if len(det) >= 7:
                x1, y1, x2, y2, track_id, cls_id, score = det[:7]
            elif len(det) >= 5:
                # Format: [x1, y1, x2, y2, score] - no tracking info
                x1, y1, x2, y2, score = det[:5]
                track_id = -1
                cls_id = 0  # Default to person
            else:
                continue

            # Scale bounding boxes if image was downsampled
            x1, y1, x2, y2 = (
                int(x1 * scale_factor),
                int(y1 * scale_factor),
                int(x2 * scale_factor),
                int(y2 * scale_factor),
            )

            # Get colour for this class
            cls_id_int = int(cls_id)
            colour = class_colours.get(cls_id_int, (128, 128, 128))

            # Draw bounding box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), colour, 2)

            # Prepare label text
            if model_names and cls_id_int < len(model_names):
                class_name = model_names[cls_id_int]
            else:
                class_name = f"Class {cls_id_int}"

            if track_id >= 0:
                label_text = f"ID:{int(track_id)} {class_name} {score:.2f}"
            else:
                label_text = f"{class_name} {score:.2f}"

            # Draw label background
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(vis_image, (x1, y1 - text_height - 10), (x1 + text_width, y1), colour, -1)
            cv2.putText(
                vis_image,
                label_text,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            num_detections += 1

    # Add info overlay
    info_text = [
        f"Frame: {frame_num}",
        f"FPS: {fps:.1f}",
        f"Detections: {num_detections}",
        "Press 'q' to quit",
    ]

    # Draw semi-transparent background for info
    overlay = vis_image.copy()
    cv2.rectangle(overlay, (10, 10), (300, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, vis_image, 0.3, 0, vis_image)

    # Draw info text
    for i, text in enumerate(info_text):
        y_pos = 35 + i * 25
        cv2.putText(vis_image, text, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return vis_image


def visualise_predictions_and_gt(
    image_np, yolo_results, target, model_names=None, max_height=1080, image_info=None, frame_num=0
):
    """
    Visualise predictions and ground truth on an image using OpenCV.

    Args:
        image_np: Image as numpy array (H, W, C) in RGB format
        yolo_results: YOLO result object
        target: Ground truth target dictionary with 'boxes' and 'labels'
        model_names: Optional class name mapping (from YOLO model)
        max_height: Maximum height for display (default: 1080p)
        image_info: Optional image info dict with 'width' and 'height' from annotations

    Returns:
        Image with visualisations as numpy array (BGR format for OpenCV)
    """
    # Get actual image dimensions
    orig_h, orig_w = image_np.shape[:2]

    # Check if annotation dimensions match image dimensions
    # If annotations are for a different resolution, we need to scale them
    annotation_scale_x = 1.0
    annotation_scale_y = 1.0

    if image_info:
        ann_width = image_info.get("width", orig_w)
        ann_height = image_info.get("height", orig_h)

        if ann_width != orig_w or ann_height != orig_h:
            # Annotations are for different resolution - scale them
            annotation_scale_x = orig_w / ann_width
            annotation_scale_y = orig_h / ann_height
            print(
                f"Warning: Annotation dimensions ({ann_width}x{ann_height}) don't match image ({orig_w}x{orig_h}). Scaling boxes by ({annotation_scale_x:.3f}, {annotation_scale_y:.3f})"
            )
    else:
        # No image_info available - assume annotations match image dimensions
        pass

    # Downsample if image is larger than max_height
    scale_factor = 1.0

    if orig_h > max_height:
        scale_factor = max_height / orig_h
        new_h = max_height
        new_w = int(orig_w * scale_factor)
        image_np = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Convert RGB to BGR for OpenCV
    vis_image = cv2.cvtColor(image_np.copy(), cv2.COLOR_RGB2BGR)

    # Draw ground truth boxes in green (dashed style - using thicker line)
    if "boxes" in target and len(target["boxes"]) > 0:
        gt_boxes = target["boxes"].cpu().numpy()
        gt_labels = (
            target.get("labels", torch.zeros(len(gt_boxes), dtype=torch.int64)).cpu().numpy()
        )

        # Debug: print first box to check coordinates (only for first frame)
        if len(gt_boxes) > 0 and frame_num == 0:
            first_box = gt_boxes[0]
            print(f"Debug Frame 0: First GT box (before scaling): {first_box}")
            print(
                f"Debug Frame 0: Image size: {orig_w}x{orig_h}, Annotation scale: ({annotation_scale_x:.3f}, {annotation_scale_y:.3f}), Display scale: {scale_factor:.3f}"
            )
            print(f"Debug Frame 0: Display image size: {vis_image.shape[1]}x{vis_image.shape[0]}")

        for box_idx, (box, label) in enumerate(zip(gt_boxes, gt_labels)):
            # First scale to match annotation dimensions, then scale for display
            x1, y1, x2, y2 = box
            x1_scaled = x1 * annotation_scale_x * scale_factor
            y1_scaled = y1 * annotation_scale_y * scale_factor
            x2_scaled = x2 * annotation_scale_x * scale_factor
            y2_scaled = y2 * annotation_scale_y * scale_factor
            x1, y1, x2, y2 = int(x1_scaled), int(y1_scaled), int(x2_scaled), int(y2_scaled)

            # Debug first box after scaling
            if box_idx == 0 and frame_num == 0:
                print(f"Debug Frame 0: First GT box (after scaling): ({x1}, {y1}, {x2}, {y2})")

            # Clamp to image bounds
            x1 = max(0, min(x1, vis_image.shape[1] - 1))
            y1 = max(0, min(y1, vis_image.shape[0] - 1))
            x2 = max(0, min(x2, vis_image.shape[1] - 1))
            y2 = max(0, min(y2, vis_image.shape[0] - 1))

            # Skip if box is invalid after clamping
            if x2 <= x1 or y2 <= y1:
                if box_idx == 0 and frame_num == 0:
                    print(
                        f"Debug Frame 0: Warning - Box became invalid after clamping: ({x1}, {y1}, {x2}, {y2})"
                    )
                continue

            # Draw dashed box (simulate with small line segments)
            dash_length = 10
            gap_length = 5
            # Top and bottom
            for x in range(x1, x2, dash_length + gap_length):
                cv2.line(vis_image, (x, y1), (min(x + dash_length, x2), y1), (0, 255, 0), 2)
                cv2.line(vis_image, (x, y2), (min(x + dash_length, x2), y2), (0, 255, 0), 2)
            # Left and right
            for y in range(y1, y2, dash_length + gap_length):
                cv2.line(vis_image, (x1, y), (x1, min(y + dash_length, y2)), (0, 255, 0), 2)
                cv2.line(vis_image, (x2, y), (x2, min(y + dash_length, y2)), (0, 255, 0), 2)

            # Add label
            if model_names and label > 0 and label <= len(model_names):
                label_name = model_names[label - 1]
            else:
                label_name = f"Class {label}"

            # Draw text background
            (text_width, text_height), baseline = cv2.getTextSize(
                f"GT: {label_name}", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                vis_image, (x1, y1 - text_height - 5), (x1 + text_width, y1), (255, 255, 255), -1
            )
            cv2.putText(
                vis_image,
                f"GT: {label_name}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

    # Draw predictions in red (solid)
    # Check if we have filtered data (from class filtering)
    if hasattr(yolo_results, "_filtered_data") and yolo_results._filtered_data is not None:
        pred_boxes = yolo_results._filtered_data["boxes"]
        pred_scores = yolo_results._filtered_data["scores"]
        pred_classes = yolo_results._filtered_data["classes"].astype(int)
    elif yolo_results.boxes is not None and len(yolo_results.boxes) > 0:
        pred_boxes = yolo_results.boxes.xyxy.cpu().numpy()
        pred_scores = yolo_results.boxes.conf.cpu().numpy()
        pred_classes = yolo_results.boxes.cls.cpu().numpy().astype(int)
    else:
        pred_boxes = None

    if pred_boxes is not None and len(pred_boxes) > 0:
        for box, score, cls in zip(pred_boxes, pred_scores, pred_classes):
            # Scale bounding boxes if image was downsampled
            x1, y1, x2, y2 = map(int, box * scale_factor)

            # Draw solid box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Add label with confidence
            if model_names and cls < len(model_names):
                label_name = model_names[cls]
            else:
                label_name = f"Class {cls}"

            label_text = f"Pred: {label_name} {score:.2f}"
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

    # Add title
    title_text = "Green (dashed): Ground Truth | Red (solid): Predictions"
    cv2.putText(vis_image, title_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return vis_image


def main():
    parser = argparse.ArgumentParser(
        description="Baseline detector (YOLOv11n) + tracker (OC-SORT) evaluation on GMIND test set"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="DeepLearning/gmind_config.yaml",
        help="Path to GMIND config YAML file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda or cpu)",
    )
    parser.add_argument(
        "--conf-threshold", type=float, default=0.25, help="Confidence threshold for detections"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="baseline_evaluation_results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--show-debug",
        action="store_true",
        help="Display debug visualisation showing predictions and ground truth",
    )
    parser.add_argument(
        "--debug-max-frames",
        type=int,
        default=10,
        help="Maximum number of frames to show in debug mode (default: 10)",
    )
    parser.add_argument(
        "--load-results",
        type=str,
        default=None,
        help="Path to existing COCO results JSON file to load (skips inference if provided)",
    )
    parser.add_argument(
        "--show-vis",
        action="store_true",
        help="Show real-time visualisation during inference (displays detections and tracking)",
    )
    parser.add_argument(
        "--save-vis",
        action="store_true",
        help="Save visualisation images for every frame (saves to output_dir/visualisations/)",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        action="store_true",
        help="Resume inference from checkpoint file if it exists",
    )
    parser.add_argument(
        "--gt-file",
        type=str,
        default=None,
        help="Path to COCO GT JSON file (if not provided, will create from dataset)",
    )
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent / config_path

    print("=" * 70)
    print("Baseline Detector + Tracker Evaluation")
    print("=" * 70)
    print(f"Config: {config_path}")
    print(f"Device: {args.device}")
    print(f"Confidence threshold: {args.conf_threshold}")
    print(f"Tracker: OC-SORT")
    print(f"Classes: person, bicycle, car (filtered)")
    print("=" * 70)

    config = load_config(config_path)

    # Extract test dataset config
    data_root = Path(config["data"]["root"])
    sensor = config["data"].get("sensor", "FLIR8.9")
    frame_stride = config["data"].get("frame_stride", 1)
    max_frames_per_video = config["data"].get("max_frames_per_video")

    test_config = config.get("test", {})
    test_sets = test_config.get("sets", [])
    test_set_subdirs = test_config.get("set_subdirs", {})
    test_percentage_split = test_config.get("percentage_split")
    test_percentage_split_start = test_config.get("percentage_split_start")

    if not test_sets:
        print("Error: No test sets specified in config")
        return

    # Load test dataset
    print(f"\nLoading test dataset: sets={test_sets}, sensor={sensor}")
    test_loader = get_gmind_dataloader(
        data_root=data_root,
        sets=test_sets,
        sensor=sensor,
        transforms=None,  # No transforms for evaluation
        batch_size=1,
        shuffle=False,
        num_workers=2,
        frame_stride=frame_stride,
        max_frames=max_frames_per_video,
        set_subdirs=test_set_subdirs,
        percentage_split=test_percentage_split,
        percentage_split_start=test_percentage_split_start,
    )
    test_dataset = test_loader.dataset

    print(
        f"Test dataset loaded: {len(test_dataset)} frames from {len(test_dataset.video_items)} videos"
    )

    # Create or load COCO ground truth (needed for evaluation)
    if args.gt_file:
        gt_path = Path(args.gt_file)
        if not gt_path.exists():
            # Try relative to output dir
            alt_path = Path(args.output_dir) / gt_path.name
            if alt_path.exists():
                gt_path = alt_path
            elif not gt_path.is_absolute():
                gt_path = Path(args.output_dir) / gt_path

        print(f"\nLoading COCO ground truth from: {gt_path}")
        from pycocotools.coco import COCO

        coco_gt = COCO(str(gt_path))

        # Get category mapping
        categories = coco_gt.loadCats(coco_gt.getCatIds())
        category_name_to_id = {cat["name"]: cat["id"] for cat in categories}

        # Create frame_idx_to_unique_image_id mapping from GT
        frame_idx_to_unique_image_id = {}
        for img_id in coco_gt.imgs.keys():
            # image_id = frame_idx + 1, so frame_idx = image_id - 1
            frame_idx = img_id - 1
            frame_idx_to_unique_image_id[frame_idx] = img_id

        temp_gt_path = None  # Not a temp file, so don't delete it
        print(f"COCO GT loaded: {len(coco_gt.imgs)} images, {len(coco_gt.anns)} annotations")
        print(f"Categories in dataset: {category_name_to_id}")
    else:
        print("\nCreating COCO ground truth...")
        coco_gt, temp_gt_path, category_name_to_id, frame_idx_to_unique_image_id = (
            create_coco_gt_from_dataset(test_dataset)
        )
        print(f"COCO GT created: {len(coco_gt.imgs)} images, {len(coco_gt.anns)} annotations")
        print(f"Categories in dataset: {category_name_to_id}")

    # Check if we should load existing results
    if args.load_results:
        # Skip model/tracker initialization when loading results
        model = None
        tracker = None
        results_path = Path(args.load_results)
        if not results_path.is_absolute():
            # Try relative to output dir first, then current dir
            results_path = Path(args.output_dir) / results_path
            if not results_path.exists():
                results_path = Path(args.load_results)

        if not results_path.exists():
            print(f"Error: Results file not found: {results_path}")
            return

        print(f"\nLoading existing results from: {results_path}")
        with open(results_path, "r") as f:
            coco_results = json.load(f)
        print(f"Loaded {len(coco_results)} detections from file")

        # Check if results use unique image IDs or overlapping ones
        # If results have image_id > number of frames, they might be using unique IDs
        # Otherwise, they likely use overlapping IDs from annotation files
        max_image_id_in_results = max([r["image_id"] for r in coco_results]) if coco_results else 0
        num_frames = len(test_dataset.frame_index)

        if max_image_id_in_results <= num_frames:
            # Results likely use unique IDs (frame_idx + 1) - good!
            print(
                f"Results appear to use unique image IDs (max: {max_image_id_in_results}, frames: {num_frames})"
            )
        else:
            print(
                f"Warning: Results have image IDs up to {max_image_id_in_results}, which exceeds frame count {num_frames}"
            )
            print(
                "This suggests results may use overlapping image IDs. Per-video metrics may be inaccurate."
            )
            print("Consider regenerating results with the updated code that uses unique image IDs.")
    else:
        # Initialize YOLOv11n
        print("\nInitializing YOLOv11n detector...")
        model = YOLO("yolo11n")  # Will download if needed
        model.to(args.device)
        print("YOLOv11n loaded successfully")

        # Initialize OC-SORT tracker
        print("\nInitializing OC-SORT tracker...")
        # Use the same confidence threshold as YOLO for OC-SORT
        tracker = OCSort(det_thresh=args.conf_threshold)
        print("OC-SORT initialized successfully")

        # Use the unique image_id mapping from COCO GT creation
        frame_to_image_id = frame_idx_to_unique_image_id.copy()

        # Check for checkpoint to resume from
        resume_frame_idx = None
        all_coco_results = []
        all_image_ids = []

        if args.resume_from_checkpoint:
            output_dir = Path(args.output_dir)
            checkpoint_file = output_dir / "coco_results_checkpoint.json"
            if checkpoint_file.exists():
                print(f"\nFound checkpoint file: {checkpoint_file}")
                with open(checkpoint_file, "r") as f:
                    checkpoint_results = json.load(f)

                if checkpoint_results:
                    # Find the highest image_id to determine resume frame
                    max_image_id = max([r["image_id"] for r in checkpoint_results])
                    # image_id = frame_idx + 1, so frame_idx = image_id - 1
                    resume_frame_idx = max_image_id - 1

                    print(f"Checkpoint contains {len(checkpoint_results)} detections")
                    print(f"Highest image_id in checkpoint: {max_image_id}")
                    print(
                        f"Resuming from frame {resume_frame_idx + 1} (batch_idx {resume_frame_idx})"
                    )

                    # Load checkpoint data
                    all_coco_results = checkpoint_results
                    # Reconstruct image_ids list from checkpoint (for tracking which frames are done)
                    image_ids_in_checkpoint = sorted(
                        set([r["image_id"] for r in checkpoint_results])
                    )
                    all_image_ids = image_ids_in_checkpoint

                    print(
                        f"Will skip frames 0-{resume_frame_idx} and continue from frame {resume_frame_idx + 1}"
                    )
                    print(f"Note: Tracker state will be reset (tracking IDs may change)")
                else:
                    print("Checkpoint file is empty, starting from beginning")
            else:
                print("No checkpoint file found, starting from beginning")

        # Run inference
        print("\nRunning inference...")
        # Store results incrementally to save memory
        # Instead of storing all YOLO result objects, convert to COCO format on-the-fly

        # For real-time visualisation
        import time

        if args.show_vis:
            fps_start_time = time.time()
            fps_frame_count = 0
            fps = 0.0

        for batch_idx, (images, targets) in enumerate(tqdm(test_loader, desc="Processing frames")):
            # Skip frames if resuming from checkpoint
            if resume_frame_idx is not None and batch_idx <= resume_frame_idx:
                # Skip this frame - already processed in checkpoint
                continue
            image = images[0]  # Batch size is 1
            target = targets[0]

            # Convert tensor to numpy for YOLO
            if isinstance(image, torch.Tensor):
                # Convert from [C, H, W] to [H, W, C] and to uint8
                image_np = image.permute(1, 2, 0).cpu().numpy()
                image_np = (image_np * 255).astype(np.uint8)
            else:
                image_np = np.array(image)

            # Run YOLO detection
            yolo_results = model.predict(
                image_np, conf=args.conf_threshold, device=args.device, verbose=False
            )

            # Apply tracker
            # Get image dimensions for OC-SORT
            img_h, img_w = image_np.shape[:2]
            img_info = np.array([img_h, img_w])  # [height, width]
            img_size = np.array([img_w, img_h])  # [width, height] for img_size

            # Filter to only person (0), bicycle (1), and car (2) classes
            # Note: YOLO boxes attributes are read-only, so we filter when extracting data
            filtered_boxes = None
            filtered_scores = None
            filtered_classes = None

            if (
                len(yolo_results) > 0
                and yolo_results[0].boxes is not None
                and len(yolo_results[0].boxes) > 0
            ):
                boxes = yolo_results[0].boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
                scores = yolo_results[0].boxes.conf.cpu().numpy()
                classes = yolo_results[0].boxes.cls.cpu().numpy()

                # Keep only person (0), bicycle (1), car (2)
                valid_mask = (classes == 0) | (classes == 1) | (classes == 2)

                if valid_mask.sum() > 0:
                    filtered_boxes = boxes[valid_mask]
                    filtered_scores = scores[valid_mask]
                    filtered_classes = classes[valid_mask]

                    # OC-SORT expects [x1, y1, x2, y2, score] format
                    detections = np.hstack([filtered_boxes, filtered_scores.reshape(-1, 1)])

                    # Update tracker
                    tracked = tracker.update(detections, img_info, img_size)

                    # Format tracked detections for visualisation: [x1, y1, x2, y2, track_id, class_id, score]
                    # OC-SORT typically returns [x1, y1, x2, y2, track_id, score] or [x1, y1, x2, y2, track_id]
                    tracked_detections = None
                    if tracked is not None and len(tracked) > 0:
                        tracked_with_class = []
                        for i, track in enumerate(tracked):
                            track = np.array(track).flatten()  # Ensure it's a flat array
                            if len(track) >= 6:
                                # Format: [x1, y1, x2, y2, track_id, score]
                                x1, y1, x2, y2, track_id, score = track[:6]
                            elif len(track) >= 5:
                                # Format: [x1, y1, x2, y2, track_id] - no score
                                x1, y1, x2, y2, track_id = track[:5]
                                score = 1.0  # Default score
                            else:
                                continue

                            # Match class by IoU or position (simplified: use position)
                            # In practice, OC-SORT doesn't preserve class info, so we'll match by box position
                            cls_id = 0  # Default
                            if i < len(filtered_classes):
                                cls_id = filtered_classes[i]
                            else:
                                # Try to match by box overlap
                                for j, (fb, fc) in enumerate(zip(filtered_boxes, filtered_classes)):
                                    # Simple center distance matching
                                    track_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
                                    box_center = np.array(
                                        [(fb[0] + fb[2]) / 2, (fb[1] + fb[3]) / 2]
                                    )
                                    if (
                                        np.linalg.norm(track_center - box_center) < 50
                                    ):  # Within 50 pixels
                                        cls_id = fc
                                        break

                            tracked_with_class.append([x1, y1, x2, y2, track_id, cls_id, score])

                        if tracked_with_class:
                            tracked_detections = np.array(tracked_with_class)
                else:
                    # No valid detections, update tracker with empty
                    tracker.update(np.empty((0, 5)), img_info, img_size)
                    tracked_detections = None
            else:
                # No detections, update tracker with empty
                tracker.update(np.empty((0, 5)), img_info, img_size)
                tracked_detections = None

            # Create filtered result for evaluation (YOLO boxes are read-only, so we'll filter in conversion)
            # Store filtered data to use later in convert_yolo_to_coco_format
            yolo_results[0]._filtered_data = (
                {"boxes": filtered_boxes, "scores": filtered_scores, "classes": filtered_classes}
                if filtered_boxes is not None
                else None
            )

            # Get image ID - use mapping to ensure consistency with COCO GT
            image_id = frame_to_image_id.get(
                batch_idx, target.get("image_id", [batch_idx + 1])[0].item()
            )

            # Real-time visualisation (shows every frame)
            if args.show_vis:
                # Calculate FPS
                fps_frame_count += 1
                if fps_frame_count % 10 == 0:  # Update FPS every 10 frames
                    elapsed = time.time() - fps_start_time
                    fps = 10.0 / elapsed if elapsed > 0 else 0.0
                    fps_start_time = time.time()

                # Get class names from YOLO model
                model_names = model.names if hasattr(model, "names") else None

                # Get image info for ground truth visualisation
                frame_info = test_dataset.frame_index[batch_idx]
                image_info = frame_info.get("image_info")

                # Create visualisation with both predictions and ground truth (like --show-debug but for all frames)
                vis_image = visualise_predictions_and_gt(
                    image_np,
                    yolo_results[0],
                    target,
                    model_names,
                    image_info=image_info,
                    frame_num=batch_idx,
                )

                # Display every frame
                try:
                    cv2.imshow("Real-time Detection & Tracking (All Frames)", vis_image)
                    # Wait 1ms to allow window to update, but don't block
                    # This allows viewing every frame in real-time
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        print("\nQuitting visualisation (inference will continue)...")
                        args.show_vis = False  # Disable further visualisation
                        cv2.destroyAllWindows()
                    elif key == ord("p"):
                        # Pause - wait for any key to continue
                        print("Paused. Press any key to continue...")
                        cv2.waitKey(0)
                except cv2.error as e:
                    # Display failed, disable visualisation
                    print(f"Display unavailable: {e}. Disabling visualisation.")
                    args.show_vis = False

            # Show debug visualisation if requested
            if args.show_debug and batch_idx < args.debug_max_frames:
                # Get class names from YOLO model
                model_names = model.names if hasattr(model, "names") else None

                # Get image info for annotation scaling
                frame_info = test_dataset.frame_index[batch_idx]
                image_info = frame_info.get("image_info")

                # Create visualisation using OpenCV
                vis_image = visualise_predictions_and_gt(
                    image_np,
                    yolo_results[0],
                    target,
                    model_names,
                    image_info=image_info,
                    frame_num=batch_idx,
                )

                # Try to display using OpenCV
                try:
                    window_name = (
                        f"Debug Visualization - Frame {batch_idx + 1}/{args.debug_max_frames}"
                    )
                    cv2.imshow(window_name, vis_image)

                    if batch_idx < args.debug_max_frames - 1:
                        # Wait briefly and then close
                        key = cv2.waitKey(500) & 0xFF  # Wait 500ms or until key press
                        if key == ord("q"):
                            print("Quitting debug visualisation...")
                            break
                        cv2.destroyWindow(window_name)
                    else:
                        # Keep last frame open until user presses a key
                        print(
                            f"\nShowing debug visualisation (frame {batch_idx + 1}/{args.debug_max_frames}). Press any key to continue..."
                        )
                        cv2.waitKey(0)  # Wait indefinitely until key press
                        cv2.destroyAllWindows()
                except cv2.error as e:
                    # OpenCV display failed (likely no display), save image instead
                    debug_output_dir = Path(args.output_dir) / "debug_visualisations"
                    debug_output_dir.mkdir(parents=True, exist_ok=True)
                    output_path = debug_output_dir / f"frame_{batch_idx:06d}.png"
                    # Convert BGR back to RGB for saving
                    vis_image_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(str(output_path), vis_image)
                    if batch_idx == 0:
                        print(
                            f"Debug visualisations saved to: {debug_output_dir} (display unavailable)"
                        )
                except Exception as e:
                    # Any other error, save image
                    debug_output_dir = Path(args.output_dir) / "debug_visualisations"
                    debug_output_dir.mkdir(parents=True, exist_ok=True)
                    output_path = debug_output_dir / f"frame_{batch_idx:06d}.png"
                    cv2.imwrite(str(output_path), vis_image)
                    if batch_idx == 0:
                        print(f"Debug visualisations saved to: {debug_output_dir} (error: {e})")

            # Convert to COCO format immediately to save memory (don't store YOLO result objects)
            # Extract detections from current frame
            frame_coco_results = []
            if (
                hasattr(yolo_results[0], "_filtered_data")
                and yolo_results[0]._filtered_data is not None
            ):
                boxes = yolo_results[0]._filtered_data["boxes"]
                scores = yolo_results[0]._filtered_data["scores"]
                classes = yolo_results[0]._filtered_data["classes"].astype(int)
            elif yolo_results[0].boxes is not None and len(yolo_results[0].boxes) > 0:
                boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
                scores = yolo_results[0].boxes.conf.cpu().numpy()
                classes = yolo_results[0].boxes.cls.cpu().numpy().astype(int)
            else:
                boxes = None

            # Convert to COCO format for this frame
            if boxes is not None and len(boxes) > 0:
                # Get class mapping (only need to build once, but it's lightweight)
                yolo_class_names = model.names if hasattr(model, "names") else {}
                yolo_cls_to_category_id = {}
                for yolo_cls_id in [0, 1, 2]:  # person, bicycle, car
                    if yolo_cls_id in yolo_class_names:
                        yolo_cls_name = yolo_class_names[yolo_cls_id].lower()
                        if yolo_cls_name in category_name_to_id:
                            yolo_cls_to_category_id[yolo_cls_id] = category_name_to_id[
                                yolo_cls_name
                            ]
                        else:
                            yolo_cls_to_category_id[yolo_cls_id] = int(yolo_cls_id + 1)

                for box, score, cls in zip(boxes, scores, classes):
                    x1, y1, x2, y2 = box
                    w = x2 - x1
                    h = y2 - y1

                    # Map YOLO class ID to dataset category ID
                    cls_int = int(cls)
                    if cls_int in yolo_cls_to_category_id:
                        category_id = yolo_cls_to_category_id[cls_int]
                    else:
                        category_id = int(cls + 1)

                    frame_coco_results.append(
                        {
                            "image_id": int(image_id),
                            "category_id": category_id,
                            "bbox": [float(x1), float(y1), float(w), float(h)],
                            "score": float(score),
                        }
                    )

            # Add to accumulated results
            all_coco_results.extend(frame_coco_results)
            all_image_ids.append(image_id)

            # Clear YOLO results from memory immediately
            del yolo_results
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Periodic checkpointing: save results every 200 frames (more frequent to avoid memory issues)
            if (batch_idx + 1) % 200 == 0:
                output_dir = Path(args.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)

                # Save checkpoint
                checkpoint_file = output_dir / "coco_results_checkpoint.json"
                with open(checkpoint_file, "w") as f:
                    json.dump(all_coco_results, f, indent=2)
                print(
                    f"\n[Checkpoint] Saved {len(all_coco_results)} detections at frame {batch_idx + 1}/{len(test_dataset)} to {checkpoint_file}"
                )

        # Results are already in COCO format
        coco_results = all_coco_results
        print(f"\nTotal detections: {len(coco_results)}")

        # Save results for future use
        # Results are saved with unique image IDs (frame_idx + 1) that match the COCO GT
        # This ensures per-video metrics work correctly and results can be reloaded later
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        coco_results_file = output_dir / "coco_results.json"
        with open(coco_results_file, "w") as f:
            json.dump(coco_results, f, indent=2)
        print(f"\n" + "=" * 70)
        print("PREDICTIONS SAVED")
        print("=" * 70)
        print(f"COCO format detections: {coco_results_file}")
        print(f"  - Contains all detections in COCO format")
        print(f"  - Format: [image_id, category_id, bbox [x,y,w,h], score]")
        print(f"  - Can be reloaded with --load-results flag")
        print(f"  - Results use unique image IDs (frame_idx + 1) for correct per-video metrics")
        print("=" * 70)

        # Remove checkpoint file if final save succeeded
        checkpoint_file = output_dir / "coco_results_checkpoint.json"
        if checkpoint_file.exists():
            try:
                checkpoint_file.unlink()
                print(f"Removed checkpoint file (final results saved)")
            except Exception:
                pass  # Ignore cleanup errors

    # Compute metrics
    print("\nComputing COCO metrics...")
    try:
        metrics, coco_eval = compute_coco_metrics(coco_gt, coco_results)
    except Exception as e:
        print(f"Error computing metrics: {e}")
        import traceback

        traceback.print_exc()
        print("\nAttempting to reload results and compute metrics...")
        # Try to reload results if there was an issue
        output_dir = Path(args.output_dir)
        coco_results_file = output_dir / "coco_results.json"
        if coco_results_file.exists():
            with open(coco_results_file, "r") as f:
                coco_results = json.load(f)
            metrics, coco_eval = compute_coco_metrics(coco_gt, coco_results)
        else:
            raise

    # Load detections for per-class evaluation (filtered results if needed)
    if len(coco_results) == 0:
        print("Warning: No detections available for per-class evaluation")
        coco_dt = None
    else:
        try:
            coco_dt = coco_gt.loadRes(coco_results)
        except Exception as e:
            print(f"Warning: Could not load detections for per-class evaluation: {e}")
            coco_dt = None

    print("\n" + "=" * 70)
    print("Overall Metrics")
    print("=" * 70)
    print(f"AP50-95: {metrics['AP50-95']:.4f}")
    print(f"AP50:    {metrics['AP50']:.4f}")
    print(f"AP75:    {metrics['AP75']:.4f}")
    print(f"APsmall: {metrics['APsmall']:.4f}")
    print(f"APlarge: {metrics['APlarge']:.4f}")

    # Per-class metrics
    print("\n" + "=" * 70)
    print("Per-Class AP50-95")
    print("=" * 70)
    if coco_dt is not None:
        per_class_ap = compute_per_class_ap(coco_gt, coco_dt)
        for class_name, ap in sorted(per_class_ap.items()):
            print(f"{class_name:20s}: {ap:.4f}")
    else:
        print("Skipping per-class metrics (no valid detections)")
        per_class_ap = {}

    # Per-video metrics
    print("\n" + "=" * 70)
    print("Per-Video Metrics")
    print("=" * 70)
    per_video_metrics = compute_per_video_metrics(
        test_dataset, coco_results, coco_gt, frame_idx_to_unique_image_id
    )
    for video_name, video_metrics in sorted(per_video_metrics.items()):
        print(f"\n{video_name}:")
        print(f"  AP50-95: {video_metrics['AP50-95']:.4f}")
        print(f"  AP50:    {video_metrics['AP50']:.4f}")
        print(f"  APsmall: {video_metrics['APsmall']:.4f}")
        print(f"  APlarge: {video_metrics['APlarge']:.4f}")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_summary = {
        "overall_metrics": metrics,
        "per_class_ap": per_class_ap,
        "per_video_metrics": per_video_metrics,
        "config": {
            "sensor": sensor,
            "test_sets": test_sets,
            "conf_threshold": args.conf_threshold,
            "tracker": "OC-SORT",
        },
    }

    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results_summary, f, indent=2)

    print(f"\n" + "=" * 70)
    print("EVALUATION RESULTS SAVED")
    print("=" * 70)
    print(f"Metrics summary: {results_file}")
    print(f"  - Contains overall metrics, per-class AP, and per-video metrics")
    print(f"  - Includes configuration used for evaluation")
    print("=" * 70)
    print(f"\nAll results saved to: {output_dir}")
    print(f"  - coco_results.json: Detection results (COCO format)")
    print(f"  - results.json: Evaluation metrics summary")
    if args.show_debug:
        print(f"  - debug_visualisations/: Debug visualisation images")

    # Cleanup temporary COCO GT file
    try:
        if os.path.exists(temp_gt_path):
            os.unlink(temp_gt_path)
    except Exception:
        pass  # Ignore cleanup errors

    print("=" * 70)


if __name__ == "__main__":
    main()
