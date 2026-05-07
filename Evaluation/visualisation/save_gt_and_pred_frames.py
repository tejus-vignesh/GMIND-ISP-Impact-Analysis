#!/usr/bin/env python3
"""
Save ground truth and prediction visualisation frames to disk as PNGs.

Renders GT + prediction bboxes onto each frame and saves them, useful for
inspecting results from cluster evaluations without needing a display.

Usage:
    python -m Evaluation.visualisation.save_gt_and_pred_frames \
        --gt-file ./sensitivity_results/coco_gt.json \
        --results ./sensitivity_results/fasterrcnn_resnet50_fpn/hsc_saturation_gain-2048/eval_results_detections.json \
        --config SensitivityAnalysis/sensitivity_config.yaml \
        --isp-variant hsc_saturation_gain-2048 \
        --output-dir ./sensitivity_results/fasterrcnn_resnet50_fpn/hsc_saturation_gain-2048/vis_frames
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from pycocotools.coco import COCO

from Evaluation.visualisation.visualise_gt_and_pred import visualise_frame


def main():
    parser = argparse.ArgumentParser(
        description="Save GT and prediction visualisation frames as PNGs"
    )
    parser.add_argument("--results", type=str, required=True, help="Path to COCO results JSON file")
    parser.add_argument("--gt-file", type=str, required=True, help="Path to COCO GT JSON file")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save PNG frames")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML (default: DeepLearning/gmind_config.yaml)",
    )
    parser.add_argument(
        "--isp-variant",
        type=str,
        default=None,
        help="ISP variant name (e.g. Default_ISP). Uses ISPVariantDataset when set.",
    )
    parser.add_argument("--score-threshold", type=float, default=0.5,
        help="Minimum confidence to display predictions (default: 0.5)")
    parser.add_argument("--start-frame", type=int, default=0, help="Start frame index (default: 0)")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to save (default: all)",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=None,
        help="Override frame_stride from config (must match the stride used during eval)",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=None,
        help="Save a single frame by image_id (e.g. 6931)",
    )
    args = parser.parse_args()

    # Load GT
    gt_path = Path(args.gt_file)
    if not gt_path.exists():
        print(f"Error: GT file not found: {gt_path}")
        sys.exit(1)

    print(f"Loading GT from: {gt_path}")
    coco_gt = COCO(str(gt_path))

    # Get category mapping
    categories = coco_gt.loadCats(coco_gt.getCatIds())
    category_id_to_name = {cat["id"]: cat["name"] for cat in categories}

    # Load results
    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Error: results file not found: {results_path}")
        sys.exit(1)

    print(f"Loading results from: {results_path}")
    with open(results_path, "r") as f:
        coco_results = json.load(f)

    # Validate results format: must be a list of dicts with required keys
    if isinstance(coco_results, dict):
        print(
            "Error: results JSON is a dict (likely a metrics summary). "
            "Expected a list of per-detection dicts with keys: "
            "image_id, bbox, score, category_id.\n"
            "Use the *_detections.json file, not the metrics results.json."
        )
        sys.exit(1)
    if not isinstance(coco_results, list):
        print(f"Error: results JSON has unexpected type: {type(coco_results).__name__}")
        sys.exit(1)
    if len(coco_results) > 0:
        required_keys = {"image_id", "bbox", "score", "category_id"}
        first = coco_results[0]
        missing = required_keys - set(first.keys())
        if missing:
            print(
                f"Error: results entries are missing keys: {missing}. "
                f"Found keys: {set(first.keys())}"
            )
            sys.exit(1)

    # Group results by image_id
    results_by_image = defaultdict(list)
    for result in coco_results:
        results_by_image[result["image_id"]].append(result)

    # Load dataset
    print("\nLoading dataset...")
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = Path(__file__).parent.parent.parent / "DeepLearning" / "gmind_config.yaml"

    if not config_path.exists():
        print(f"Error: config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data_root = Path(config["data"]["root"])
    sensor = config["data"].get("sensor", "FLIR8.9")
    frame_stride = args.frame_stride if args.frame_stride is not None else config["data"].get("frame_stride", 1)
    max_frames_per_video = config["data"].get("max_frames_per_video")

    test_config = config.get("test", {})
    test_sets = test_config.get("sets", [])
    test_set_subdirs = test_config.get("set_subdirs", {})
    test_percentage_split = test_config.get("percentage_split", {})
    test_percentage_split_start = test_config.get("percentage_split_start", {})

    if args.isp_variant:
        from SensitivityAnalysis.isp_dataset import get_isp_dataloader

        test_loader = get_isp_dataloader(
            data_root=data_root,
            isp_variant=args.isp_variant,
            sets=test_sets,
            sensor=sensor,
            transforms=None,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            frame_stride=frame_stride,
            max_frames=max_frames_per_video,
            set_subdirs=test_set_subdirs,
            percentage_split=test_percentage_split,
            percentage_split_start=test_percentage_split_start,
        )
    else:
        from DataLoader import get_gmind_dataloader

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
            percentage_split=test_percentage_split,
            percentage_split_start=test_percentage_split_start,
        )
    test_dataset = test_loader.dataset

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine frame range
    end_frame = len(test_dataset)
    if args.max_frames is not None:
        end_frame = min(end_frame, args.start_frame + args.max_frames)

    total_to_save = end_frame - args.start_frame
    print(f"Dataset loaded: {len(test_dataset)} frames")
    print(f"Saving frames {args.start_frame} to {end_frame - 1} ({total_to_save} frames)")
    print(f"Output directory: {output_dir}\n")

    saved_count = 0
    for idx in range(args.start_frame, end_frame):
        if idx >= len(test_dataset):
            break

        # Get frame
        images, targets = test_dataset[idx]
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

        # Get image_id from dataset target
        image_id = (
            int(target["image_id"].item())
            if isinstance(target["image_id"], torch.Tensor)
            else int(target["image_id"])
        )

        # Single-frame mode: skip non-matching image_ids
        if args.frame is not None and image_id != args.frame:
            continue

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

        # Filter by score threshold
        pred_boxes = [b for b in pred_boxes if b[5] >= args.score_threshold]

        # Render visualisation
        vis_image = visualise_frame(image_np, gt_boxes, pred_boxes, category_id_to_name)

        # Add frame info overlay at the top
        frame_text = (
            f"Frame {idx + 1}/{len(test_dataset)} | "
            f"Ground Truths: {len(gt_boxes)} | Predictions: {len(pred_boxes)}"
        )
        cv2.putText(
            vis_image,
            frame_text,
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 255),
            3,
        )

        # Save frame
        out_path = output_dir / f"frame_{image_id:06d}.png"
        cv2.imwrite(str(out_path), vis_image)
        saved_count += 1

        # Single-frame mode: done after saving the match
        if args.frame is not None:
            break

        # Progress every 50 frames
        if saved_count % 50 == 0:
            print(f"  Saved {saved_count}/{total_to_save} frames...")

    print(f"\nDone. Saved {saved_count} frames to {output_dir}")


if __name__ == "__main__":
    main()
