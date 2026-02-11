"""Export GMIND dataset to YOLO format for Ultralytics training.

Optimized version: keeps VideoCapture open per video and reads frames
sequentially instead of reopening and seeking for every frame.
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import yaml
from PIL import Image

logger = logging.getLogger(__name__)


def export_gmind_to_yolo(
    train_dataset, val_dataset, out_dir: str, class_names: Optional[List[str]] = None
) -> str:
    """Export GMIND datasets to Ultralytics YOLO format.

    Args:
        train_dataset: GMINDDataset instance for training
        val_dataset: GMINDDataset instance for validation
        out_dir: Output directory where train/val folders will be created
        class_names: Optional list of class names (will be inferred from dataset if not provided)

    Returns:
        Path to the generated data.yaml file

    Example:
        >>> from DataLoader import GMINDDataset, export_gmind_to_yolo
        >>> train_dataset = GMINDDataset(...)
        >>> val_dataset = GMINDDataset(...)
        >>> data_yaml = export_gmind_to_yolo(train_dataset, val_dataset, "/path/to/output")
    """
    from . import GMINDDataset

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    train_images_dir = out_path / "train" / "images"
    train_labels_dir = out_path / "train" / "labels"
    val_images_dir = out_path / "val" / "images"
    val_labels_dir = out_path / "val" / "labels"

    for d in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Get class names from dataset if not provided
    if class_names is None:
        class_names = _infer_class_names(train_dataset)
        logger.info(f"Inferred {len(class_names)} classes from dataset")

    # Export training set
    logger.info(f"Exporting {len(train_dataset)} training images...")
    _export_dataset_split(train_dataset, train_images_dir, train_labels_dir, class_names)

    # Export validation set
    logger.info(f"Exporting {len(val_dataset)} validation images...")
    _export_dataset_split(val_dataset, val_images_dir, val_labels_dir, class_names)

    # Create data.yaml
    data_yaml = out_path / "data.yaml"
    data = {
        "path": str(out_path.absolute()),
        "train": "train/images",
        "val": "val/images",
        "nc": len(class_names),
        "names": class_names,
    }

    with open(data_yaml, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    logger.info(f"YOLO format export complete: {data_yaml}")
    logger.info(f"  Training images: {len(list(train_images_dir.glob('*.jpg')))}")
    logger.info(f"  Validation images: {len(list(val_images_dir.glob('*.jpg')))}")
    logger.info(f"  Classes: {len(class_names)}")

    return str(data_yaml)


def _infer_class_names(dataset) -> List[str]:
    """Infer class names from GMIND dataset annotations."""
    class_names = []
    class_ids = set()

    # Try to get categories from annotation files
    try:
        # Look at the first video item to get annotation path
        if hasattr(dataset, "video_items") and len(dataset.video_items) > 0:
            ann_path = dataset.video_items[0].get("annotation_path")
            if ann_path and Path(ann_path).exists():
                with open(ann_path, "r") as f:
                    ann_data = json.load(f)
                categories = ann_data.get("categories", [])
                if categories:
                    # Sort by id and extract names
                    sorted_cats = sorted(categories, key=lambda x: x["id"])
                    class_names = [cat["name"] for cat in sorted_cats]
                    class_ids = {cat["id"] for cat in categories}
                    logger.info(f"Found {len(class_names)} classes in annotations")
    except Exception as e:
        logger.warning(f"Could not infer class names from annotations: {e}")

    # If no classes found, use defaults
    if not class_names:
        class_names = ["person", "bicycle", "car", "background"]
        logger.info(f"Using default class names: {class_names}")

    return class_names


def _export_dataset_split(dataset, images_dir: Path, labels_dir: Path, class_names: List[str]):
    """Export a dataset split (train or val) to YOLO format.

    Optimized: groups frames by video and reads sequentially to avoid
    repeated open/seek/close overhead per frame.
    """
    from . import GMINDDataset

    # Create mapping from COCO category IDs to YOLO class indices
    category_id_to_class_idx = {}

    # Try to get category mapping from first annotation file
    try:
        if hasattr(dataset, "video_items") and len(dataset.video_items) > 0:
            ann_path = dataset.video_items[0].get("annotation_path")
            if ann_path and Path(ann_path).exists():
                with open(ann_path, "r") as f:
                    ann_data = json.load(f)
                categories = ann_data.get("categories", [])
                for cat in categories:
                    # YOLO uses 0-based indexing, COCO uses 1-based
                    category_id_to_class_idx[cat["id"]] = cat["id"] - 1
    except Exception as e:
        logger.warning(f"Could not get category mapping: {e}")
        # Default mapping: assume sequential IDs starting from 1
        for i, name in enumerate(class_names):
            category_id_to_class_idx[i + 1] = i

    # Group frame_index entries by video, preserving the global dataset idx
    frames_by_video = defaultdict(list)
    for dataset_idx, item in enumerate(dataset.frame_index):
        frames_by_video[item["video_idx"]].append((dataset_idx, item))

    exported_count = 0
    for video_idx in sorted(frames_by_video.keys()):
        video_item = dataset.video_items[video_idx]
        video_path = video_item["video_path"]

        # Build lookup: frame_number -> (dataset_idx, item)
        entries = frames_by_video[video_idx]
        needed = {item["frame_idx"]: (ds_idx, item) for ds_idx, item in entries}
        max_frame = max(needed.keys())

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.warning(f"Could not open video {video_path}")
            continue

        current_frame = 0
        while current_frame <= max_frame:
            ret, frame = cap.read()
            if not ret:
                logger.warning(
                    f"Failed to read frame {current_frame} from {video_path}"
                )
                break

            if current_frame in needed:
                ds_idx, item = needed[current_frame]

                # Convert BGR -> RGB -> PIL
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)

                # Save image
                img_name = f"{ds_idx:06d}.jpg"
                frame_pil.save(images_dir / img_name, quality=95)

                # Convert annotations to YOLO format
                h, w = frame_pil.size[1], frame_pil.size[0]
                label_lines = []
                annotations = item.get("annotations", [])

                for ann in annotations:
                    # Get bounding box (COCO format: [x, y, width, height])
                    bbox = ann.get("bbox", [])
                    if len(bbox) != 4:
                        continue

                    x, y, w_box, h_box = bbox

                    # Convert to YOLO format: normalized center_x, center_y, width, height
                    center_x = (x + w_box / 2.0) / w
                    center_y = (y + h_box / 2.0) / h
                    norm_w = w_box / w
                    norm_h = h_box / h

                    # Get class ID
                    category_id = ann.get("category_id", 0)
                    class_idx = category_id_to_class_idx.get(
                        category_id, category_id - 1
                    )

                    # Clamp values to [0, 1]
                    center_x = max(0, min(1, center_x))
                    center_y = max(0, min(1, center_y))
                    norm_w = max(0, min(1, norm_w))
                    norm_h = max(0, min(1, norm_h))

                    label_lines.append(
                        f"{class_idx} {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}"
                    )

                # Save labels
                label_path = labels_dir / (img_name.replace(".jpg", ".txt"))
                with open(label_path, "w") as f:
                    f.write("\n".join(label_lines))

                exported_count += 1

                if exported_count % 100 == 0:
                    logger.info(
                        f"  Exported {exported_count}/{len(dataset)} images..."
                    )

            current_frame += 1

        cap.release()

    logger.info(f"Exported {exported_count}/{len(dataset)} images to {images_dir}")
    