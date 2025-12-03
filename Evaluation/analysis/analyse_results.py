#!/usr/bin/env python3
"""
Unified analysis script for evaluation results.

This script provides comprehensive analysis capabilities for object detection
and tracking evaluation results. It combines functionality from multiple
specialised analysis scripts into a single, flexible tool.

Available Analyses:
    --detection: General detection analysis (objects per frame, class breakdown,
                 GT overlaps, detection quality metrics)
    --missed: Missed objects per frame analysis (especially large objects)
    --mota: MOTA (Multiple Object Tracking Accuracy) per video breakdown
    --overall-mota: Compute overall MOTA for the entire dataset
    --size-plot: Plot object size vs miss rate and AP50
    --large: Debug analysis for large object detection performance
    --full-report: Comprehensive evaluation report with all metrics
    --compare-static-moving: Compare metrics for static vs moving objects

Usage Examples:
    # Run all analyses
    python -m Evaluation.analysis.analyse_results --all --output-dir results/

    # Run specific analyses
    python -m Evaluation.analysis.analyse_results --detection --missed

    # Generate comprehensive report
    python -m Evaluation.analysis.analyse_results --full-report

    # Compare static vs moving objects
    python -m Evaluation.analysis.analyse_results --compare-static-moving

Input Format:
    - Ground truth: COCO format JSON file
    - Results: COCO results format JSON file (list of detection dicts)
    - Each detection dict must have: image_id, category_id, bbox, score

Output:
    All analyses print results to stdout. Some analyses may create additional
    files in the output directory (e.g., plots, summary tables).
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from Evaluation.analysis.analysis_utils import (
    compute_coco_metrics,
    compute_iou,
    compute_mota_simple,
    compute_per_class_ap,
    get_area_category,
    get_video_image_mapping,
    group_by_image,
    match_predictions_to_gt,
)
from Evaluation.core.baseline_detector_and_tracker import (
    create_coco_gt_from_dataset,
    get_gmind_dataloader,
    load_config,
)

# ============================================================================
# Analysis Functions
# ============================================================================


def check_gt_overlaps(coco_gt, category_name_to_id):
    """
    Check for overlapping ground truth annotations of the same class.

    Identifies cases where multiple GT annotations of the same class overlap
    in the same image. This can indicate annotation issues or legitimate
    cases (e.g., occluded objects).

    Args:
        coco_gt: COCO ground truth object
        category_name_to_id: Dictionary mapping category names to IDs

    Returns:
        Tuple of (overlaps_by_class, overlap_stats):
        - overlaps_by_class: Dict mapping class name to list of overlap dicts
        - overlap_stats: Dict mapping class name to statistics (count, pairs, max_iou, mean_iou)
    """
    cat_id_to_name = {v: k for k, v in category_name_to_id.items()}

    overlaps_by_class = defaultdict(list)
    overlap_stats = defaultdict(lambda: {"count": 0, "pairs": 0, "max_iou": 0.0, "mean_iou": []})

    print("\nChecking for overlapping GT annotations...")

    for img_id in coco_gt.imgs.keys():
        ann_ids = coco_gt.getAnnIds(imgIds=[img_id])
        anns = coco_gt.loadAnns(ann_ids)

        # Group by category
        anns_by_class = defaultdict(list)
        for ann in anns:
            cat_id = ann["category_id"]
            cat_name = cat_id_to_name.get(cat_id, f"unknown_{cat_id}")
            anns_by_class[cat_name].append(ann)

        # Check overlaps within each class
        for cat_name, class_anns in anns_by_class.items():
            if len(class_anns) < 2:
                continue

            # Check all pairs
            for i in range(len(class_anns)):
                for j in range(i + 1, len(class_anns)):
                    ann1 = class_anns[i]
                    ann2 = class_anns[j]

                    iou = compute_iou(ann1["bbox"], ann2["bbox"])
                    if iou > 0.0:
                        overlaps_by_class[cat_name].append(
                            {
                                "img_id": img_id,
                                "ann1_id": ann1["id"],
                                "ann2_id": ann2["id"],
                                "iou": iou,
                                "bbox1": ann1["bbox"],
                                "bbox2": ann2["bbox"],
                            }
                        )
                        overlap_stats[cat_name]["count"] += 1
                        overlap_stats[cat_name]["pairs"] += 1
                        overlap_stats[cat_name]["mean_iou"].append(iou)
                        overlap_stats[cat_name]["max_iou"] = max(
                            overlap_stats[cat_name]["max_iou"], iou
                        )

    return overlaps_by_class, overlap_stats


def analyse_detection_results(coco_gt, coco_results, category_name_to_id):
    """
    Perform comprehensive detection results analysis.

    Analyses include:
    - Objects per frame statistics (GT vs predictions)
    - Class breakdown (number of instances per class)
    - Ground truth overlap analysis
    - Detection quality metrics (score distributions)
    - Frame-level analysis (frames with zero detections, etc.)

    Args:
        coco_gt: COCO ground truth object
        coco_results: List of prediction dictionaries in COCO results format
        category_name_to_id: Dictionary mapping category names to IDs

    Prints:
        Detailed statistics and analysis results to stdout.
    """
    # Get category mappings
    cat_id_to_name = {v: k for k, v in category_name_to_id.items()}

    # Check for overlapping GT annotations
    overlaps_by_class, overlap_stats = check_gt_overlaps(coco_gt, category_name_to_id)

    # Statistics
    gt_per_frame = defaultdict(int)
    pred_per_frame = defaultdict(int)
    gt_per_class = Counter()
    pred_per_class = Counter()

    # Analyse ground truth
    print("Analysing ground truth...")
    for img_id in coco_gt.imgs.keys():
        ann_ids = coco_gt.getAnnIds(imgIds=[img_id])
        anns = coco_gt.loadAnns(ann_ids)

        gt_per_frame[img_id] = len(anns)
        for ann in anns:
            cat_id = ann["category_id"]
            cat_name = cat_id_to_name.get(cat_id, f"unknown_{cat_id}")
            gt_per_class[cat_name] += 1

    # Analyse predictions
    print("Analysing predictions...")
    for result in coco_results:
        img_id = result["image_id"]
        cat_id = result["category_id"]
        cat_name = cat_id_to_name.get(cat_id, f"unknown_{cat_id}")

        pred_per_frame[img_id] += 1
        pred_per_class[cat_name] += 1

    # Calculate statistics
    gt_counts = list(gt_per_frame.values())
    pred_counts = list(pred_per_frame.values())

    # Objects per frame statistics
    print("\n" + "=" * 70)
    print("OBJECTS PER FRAME STATISTICS")
    print("=" * 70)
    print(f"Ground Truth:")
    print(f"  Total frames: {len(gt_per_frame)}")
    print(f"  Total objects: {sum(gt_counts)}")
    print(f"  Mean objects/frame: {np.mean(gt_counts):.2f}")
    print(f"  Median objects/frame: {np.median(gt_counts):.2f}")
    print(f"  Min objects/frame: {np.min(gt_counts)}")
    print(f"  Max objects/frame: {np.max(gt_counts)}")
    print(f"  Std dev: {np.std(gt_counts):.2f}")

    print(f"\nPredictions:")
    print(f"  Total frames with detections: {len(pred_per_frame)}")
    print(f"  Total detections: {sum(pred_counts)}")
    print(f"  Mean detections/frame: {np.mean(pred_counts) if pred_counts else 0:.2f}")
    print(f"  Median detections/frame: {np.median(pred_counts) if pred_counts else 0:.2f}")
    print(f"  Min detections/frame: {np.min(pred_counts) if pred_counts else 0}")
    print(f"  Max detections/frame: {np.max(pred_counts) if pred_counts else 0}")
    print(f"  Std dev: {np.std(pred_counts) if pred_counts else 0:.2f}")

    # Compare per frame
    all_img_ids = set(gt_per_frame.keys()) | set(pred_per_frame.keys())
    frame_diffs = []
    for img_id in sorted(all_img_ids):
        gt_count = gt_per_frame.get(img_id, 0)
        pred_count = pred_per_frame.get(img_id, 0)
        diff = pred_count - gt_count
        frame_diffs.append(diff)

    print(f"\nPer-Frame Comparison:")
    print(f"  Frames with more predictions than GT: {sum(1 for d in frame_diffs if d > 0)}")
    print(f"  Frames with fewer predictions than GT: {sum(1 for d in frame_diffs if d < 0)}")
    print(f"  Frames with equal counts: {sum(1 for d in frame_diffs if d == 0)}")
    print(f"  Mean difference (pred - GT): {np.mean(frame_diffs):.2f}")
    print(f"  Median difference: {np.median(frame_diffs):.2f}")

    # GT Overlap Analysis
    print("\n" + "=" * 70)
    print("GROUND TRUTH OVERLAP ANALYSIS (Same Class)")
    print("=" * 70)
    if any(overlap_stats.values()):
        print(
            f"{'Class':<15} {'Overlap Pairs':<15} {'Max IoU':<12} {'Mean IoU':<12} {'Affected Images':<15}"
        )
        print("-" * 70)
        for cat_name in sorted(overlap_stats.keys()):
            stats = overlap_stats[cat_name]
            if stats["pairs"] > 0:
                affected_images = len(set([o["img_id"] for o in overlaps_by_class[cat_name]]))
                mean_iou = np.mean(stats["mean_iou"]) if stats["mean_iou"] else 0.0
                print(
                    f"{cat_name:<15} {stats['pairs']:<15} {stats['max_iou']:.4f}      {mean_iou:.4f}      {affected_images:<15}"
                )

        # Show some examples
        print("\nExample overlapping pairs (first 10):")
        example_count = 0
        for cat_name in sorted(overlaps_by_class.keys()):
            for overlap in overlaps_by_class[cat_name][:5]:
                if example_count >= 10:
                    break
                print(f"  {cat_name} - Image {overlap['img_id']}: IoU={overlap['iou']:.4f}")
                example_count += 1
            if example_count >= 10:
                break
    else:
        print("No overlapping GT annotations found for same class.")

    # Class breakdown
    print("\n" + "=" * 70)
    print("CLASS BREAKDOWN (Number of Instances)")
    print("=" * 70)
    all_classes = set(gt_per_class.keys()) | set(pred_per_class.keys())

    print(f"{'Class':<15} {'GT Count':<12} {'Pred Count':<12} {'Difference':<12} {'Ratio':<10}")
    print("-" * 70)
    for class_name in sorted(all_classes):
        gt_count = gt_per_class.get(class_name, 0)
        pred_count = pred_per_class.get(class_name, 0)
        diff = pred_count - gt_count
        ratio = pred_count / gt_count if gt_count > 0 else float("inf") if pred_count > 0 else 0.0
        print(f"{class_name:<15} {gt_count:<12} {pred_count:<12} {diff:<12} {ratio:.2f}")

    total_gt = sum(gt_per_class.values())
    total_pred = sum(pred_per_class.values())
    print("-" * 70)
    print(
        f"{'TOTAL':<15} {total_gt:<12} {total_pred:<12} {total_pred - total_gt:<12} {total_pred/total_gt if total_gt > 0 else 0:.2f}"
    )

    # Detection quality by score
    print("\n" + "=" * 70)
    print("DETECTION QUALITY (Score Distribution)")
    print("=" * 70)
    all_scores = [r["score"] for r in coco_results]
    if all_scores:
        print(f"  Total detections: {len(all_scores)}")
        print(f"  Mean score: {np.mean(all_scores):.4f}")
        print(f"  Median score: {np.median(all_scores):.4f}")
        print(f"  Min score: {np.min(all_scores):.4f}")
        print(f"  Max score: {np.max(all_scores):.4f}")
        print(
            f"  Score < 0.3: {sum(1 for s in all_scores if s < 0.3)} ({100*sum(1 for s in all_scores if s < 0.3)/len(all_scores):.1f}%)"
        )
        print(
            f"  Score < 0.5: {sum(1 for s in all_scores if s < 0.5)} ({100*sum(1 for s in all_scores if s < 0.5)/len(all_scores):.1f}%)"
        )
        print(
            f"  Score >= 0.5: {sum(1 for s in all_scores if s >= 0.5)} ({100*sum(1 for s in all_scores if s >= 0.5)/len(all_scores):.1f}%)"
        )
        print(
            f"  Score >= 0.7: {sum(1 for s in all_scores if s >= 0.7)} ({100*sum(1 for s in all_scores if s >= 0.7)/len(all_scores):.1f}%)"
        )

    # Per-class score statistics
    print("\nPer-Class Score Statistics:")
    for class_name in sorted(pred_per_class.keys()):
        class_scores = [
            r["score"]
            for r in coco_results
            if cat_id_to_name.get(r["category_id"], "") == class_name
        ]
        if class_scores:
            print(f"  {class_name}:")
            print(f"    Count: {len(class_scores)}")
            print(f"    Mean score: {np.mean(class_scores):.4f}")
            print(f"    Median score: {np.median(class_scores):.4f}")

    # Frames with zero detections
    frames_with_gt = set(gt_per_frame.keys())
    frames_with_pred = set(pred_per_frame.keys())
    frames_no_pred = frames_with_gt - frames_with_pred
    frames_no_gt = frames_with_pred - frames_with_gt

    print("\n" + "=" * 70)
    print("FRAME-LEVEL ANALYSIS")
    print("=" * 70)
    print(
        f"Frames with GT but no predictions: {len(frames_no_pred)} ({100*len(frames_no_pred)/len(frames_with_gt):.1f}%)"
    )
    print(f"Frames with predictions but no GT: {len(frames_no_gt)}")

    # Show some examples of frames with large differences
    print("\nFrames with largest prediction-GT differences:")
    frame_diffs_list = [
        (
            img_id,
            pred_per_frame.get(img_id, 0) - gt_per_frame.get(img_id, 0),
            gt_per_frame.get(img_id, 0),
            pred_per_frame.get(img_id, 0),
        )
        for img_id in all_img_ids
    ]
    frame_diffs_list.sort(key=lambda x: abs(x[1]), reverse=True)

    print(f"{'Image ID':<12} {'GT':<8} {'Pred':<8} {'Diff':<8}")
    print("-" * 40)
    for img_id, diff, gt, pred in frame_diffs_list[:10]:
        print(f"{img_id:<12} {gt:<8} {pred:<8} {diff:<8}")


def analyse_missed_per_frame(coco_gt, coco_results):
    """
    Analyse distribution of missed objects across frames.

    Focuses on understanding when and where objects are missed, with special
    attention to large objects which are often more critical to detect.

    Analyses include:
    - Missed objects per frame (all objects and large objects only)
    - Distribution statistics (mean, median, max missed per frame)
    - Per-category breakdown of missed objects
    - Frames with zero missed objects

    Args:
        coco_gt: COCO ground truth object
        coco_results: List of prediction dictionaries in COCO results format

    Prints:
        Detailed missed object statistics to stdout.
    """
    # Get category mapping
    categories = coco_gt.loadCats(coco_gt.getCatIds())
    cat_id_to_name = {cat["id"]: cat["name"] for cat in categories}

    # Group GT and predictions by image_id
    gt_by_image, pred_by_image = group_by_image(coco_gt, coco_results)

    # Analyse per frame
    missed_per_frame = []
    missed_large_per_frame = []
    missed_by_category_per_frame = defaultdict(list)
    missed_large_by_category_per_frame = defaultdict(list)

    total_gt_per_frame = []
    total_pred_per_frame = []

    for img_id in sorted(coco_gt.imgs.keys()):
        gt_anns = gt_by_image.get(img_id, [])
        pred_results = pred_by_image.get(img_id, [])

        total_gt_per_frame.append(len(gt_anns))
        total_pred_per_frame.append(len(pred_results))

        # Match predictions to GT
        matched_gt_ids, _ = match_predictions_to_gt(pred_results, gt_anns, iou_threshold=0.5)

        # Count missed objects
        missed_count = 0
        missed_large_count = 0
        missed_by_cat = Counter()
        missed_large_by_cat = Counter()

        for gt_ann in gt_anns:
            if gt_ann["id"] not in matched_gt_ids:
                missed_count += 1
                cat_name = cat_id_to_name.get(gt_ann["category_id"], "unknown")
                missed_by_cat[cat_name] += 1

                area = gt_ann.get("area", gt_ann["bbox"][2] * gt_ann["bbox"][3])
                if get_area_category(area) == "large":
                    missed_large_count += 1
                    missed_large_by_cat[cat_name] += 1

        missed_per_frame.append(missed_count)
        missed_large_per_frame.append(missed_large_count)

        for cat_name, count in missed_by_cat.items():
            missed_by_category_per_frame[cat_name].append(count)

        for cat_name, count in missed_large_by_cat.items():
            missed_large_by_category_per_frame[cat_name].append(count)

    # Print statistics
    print("\n" + "=" * 70)
    print("MISSED OBJECTS PER FRAME - ALL OBJECTS")
    print("=" * 70)
    print(f"Total frames: {len(missed_per_frame)}")
    print(f"Total missed objects: {sum(missed_per_frame)}")
    print(f"Mean missed per frame: {np.mean(missed_per_frame):.2f}")
    print(f"Median missed per frame: {np.median(missed_per_frame):.2f}")
    print(f"Min missed per frame: {np.min(missed_per_frame)}")
    print(f"Max missed per frame: {np.max(missed_per_frame)}")
    print(f"Std dev: {np.std(missed_per_frame):.2f}")

    # Distribution
    missed_counts = Counter(missed_per_frame)
    print(f"\nDistribution of missed objects per frame:")
    print(f"{'Missed Count':<15} {'Frames':<15} {'Percentage':<15}")
    print("-" * 45)
    for missed_count in sorted(missed_counts.keys())[:20]:  # Show first 20
        frame_count = missed_counts[missed_count]
        percentage = 100 * frame_count / len(missed_per_frame)
        print(f"{missed_count:<15} {frame_count:<15} {percentage:.1f}%")
    if len(missed_counts) > 20:
        print(f"... ({len(missed_counts) - 20} more unique values)")

    # Frames with most missed objects
    print(f"\nFrames with most missed objects (top 10):")
    frame_missed = [(i + 1, missed_per_frame[i]) for i in range(len(missed_per_frame))]
    frame_missed.sort(key=lambda x: x[1], reverse=True)
    print(f"{'Frame':<10} {'Missed':<10} {'GT Total':<10} {'Pred Total':<10}")
    print("-" * 40)
    for frame_idx, missed_count in frame_missed[:10]:
        gt_total = total_gt_per_frame[frame_idx - 1]
        pred_total = (
            total_pred_per_frame[frame_idx - 1] if frame_idx - 1 < len(total_pred_per_frame) else 0
        )
        print(f"{frame_idx:<10} {missed_count:<10} {gt_total:<10} {pred_total:<10}")

    # Large objects
    print("\n" + "=" * 70)
    print("MISSED LARGE OBJECTS PER FRAME")
    print("=" * 70)
    print(f"Total frames: {len(missed_large_per_frame)}")
    print(f"Total missed large objects: {sum(missed_large_per_frame)}")
    print(f"Mean missed large per frame: {np.mean(missed_large_per_frame):.2f}")
    print(f"Median missed large per frame: {np.median(missed_large_per_frame):.2f}")
    print(f"Min missed large per frame: {np.min(missed_large_per_frame)}")
    print(f"Max missed large per frame: {np.max(missed_large_per_frame)}")
    print(f"Std dev: {np.std(missed_large_per_frame):.2f}")

    # Distribution for large
    missed_large_counts = Counter(missed_large_per_frame)
    print(f"\nDistribution of missed large objects per frame:")
    print(f"{'Missed Large':<15} {'Frames':<15} {'Percentage':<15}")
    print("-" * 45)
    for missed_count in sorted(missed_large_counts.keys())[:20]:
        frame_count = missed_large_counts[missed_count]
        percentage = 100 * frame_count / len(missed_large_per_frame)
        print(f"{missed_count:<15} {frame_count:<15} {percentage:.1f}%")
    if len(missed_large_counts) > 20:
        print(f"... ({len(missed_large_counts) - 20} more unique values)")

    # Frames with most missed large objects
    print(f"\nFrames with most missed large objects (top 10):")
    frame_missed_large = [
        (i + 1, missed_large_per_frame[i]) for i in range(len(missed_large_per_frame))
    ]
    frame_missed_large.sort(key=lambda x: x[1], reverse=True)
    print(f"{'Frame':<10} {'Missed Large':<15} {'GT Total':<10} {'Pred Total':<10}")
    print("-" * 50)
    for frame_idx, missed_count in frame_missed_large[:10]:
        gt_total = total_gt_per_frame[frame_idx - 1]
        pred_total = (
            total_pred_per_frame[frame_idx - 1] if frame_idx - 1 < len(total_pred_per_frame) else 0
        )
        print(f"{frame_idx:<10} {missed_count:<15} {gt_total:<10} {pred_total:<10}")

    # Per-category analysis
    print("\n" + "=" * 70)
    print("MISSED OBJECTS PER FRAME BY CATEGORY")
    print("=" * 70)
    for cat_name in sorted(missed_by_category_per_frame.keys()):
        missed_counts = missed_by_category_per_frame[cat_name]
        print(f"\n{cat_name}:")
        print(f"  Total missed: {sum(missed_counts)}")
        print(f"  Mean per frame: {np.mean(missed_counts):.2f}")
        print(f"  Median per frame: {np.median(missed_counts):.2f}")
        print(f"  Max per frame: {np.max(missed_counts)}")

    print("\n" + "=" * 70)
    print("MISSED LARGE OBJECTS PER FRAME BY CATEGORY")
    print("=" * 70)
    for cat_name in sorted(missed_large_by_category_per_frame.keys()):
        missed_counts = missed_large_by_category_per_frame[cat_name]
        print(f"\n{cat_name}:")
        print(f"  Total missed large: {sum(missed_counts)}")
        print(f"  Mean per frame: {np.mean(missed_counts):.2f}")
        print(f"  Median per frame: {np.median(missed_counts):.2f}")
        print(f"  Max per frame: {np.max(missed_counts)}")

    # Frames with zero missed objects
    frames_zero_missed = sum(1 for m in missed_per_frame if m == 0)
    frames_zero_missed_large = sum(1 for m in missed_large_per_frame if m == 0)

    print("\n" + "=" * 70)
    print("FRAMES WITH ZERO MISSED OBJECTS")
    print("=" * 70)
    print(
        f"Frames with zero missed objects: {frames_zero_missed} ({100*frames_zero_missed/len(missed_per_frame):.1f}%)"
    )
    print(
        f"Frames with zero missed large objects: {frames_zero_missed_large} ({100*frames_zero_missed_large/len(missed_large_per_frame):.1f}%)"
    )


def compute_mota_per_video(coco_gt, coco_results, test_dataset, frame_idx_to_unique_image_id):
    """
    Compute MOTA per video with detailed breakdown.

    Computes MOTA separately for each video in the dataset, allowing
    identification of videos where the model performs better or worse.

    Args:
        coco_gt: COCO ground truth object
        coco_results: List of prediction dictionaries in COCO results format
        test_dataset: GMIND test dataset object
        frame_idx_to_unique_image_id: Mapping from dataset frame index to COCO image ID

    Returns:
        Dictionary mapping video names to statistics dicts containing:
        - "total_gt", "total_pred", "matched", "false_negatives", "false_positives"
        - "MOTA", "precision", "recall"
        - "matched_by_class", "missed_by_class", "fp_by_class"
    """
    # Group GT and predictions by image for efficient per-frame matching
    gt_by_image, pred_by_image = group_by_image(coco_gt, coco_results)

    # Map image IDs to videos (needed for per-video aggregation)
    video_image_ids = get_video_image_mapping(test_dataset, frame_idx_to_unique_image_id)

    per_video_stats = {}

    # Process each video separately
    for video_name, img_ids in video_image_ids.items():
        # Get GT and predictions for this video
        video_gt_anns = []
        for ann in coco_gt.anns.values():
            if ann["image_id"] in img_ids:
                video_gt_anns.append(ann)

        video_results = [r for r in coco_results if r["image_id"] in img_ids]

        # Initialise tracking variables for this video
        matched_gt = set()  # Set of matched GT annotation IDs
        false_positives = 0  # Count of unmatched predictions
        matched_by_class = defaultdict(int)  # Matched objects per class
        missed_by_class = defaultdict(int)  # Missed GT objects per class
        fp_by_class = defaultdict(int)  # False positives per class

        # Group annotations by image for efficient per-frame processing
        video_gt_by_image = defaultdict(list)
        video_pred_by_image = defaultdict(list)

        for ann in video_gt_anns:
            video_gt_by_image[ann["image_id"]].append(ann)

        for result in video_results:
            video_pred_by_image[result["image_id"]].append(result)

        # Process each frame in this video
        for img_id in sorted(img_ids):
            if img_id not in coco_gt.imgs:
                continue

            gt_anns = video_gt_by_image.get(img_id, [])
            pred_results = video_pred_by_image.get(img_id, [])

            # Match predictions to GT using IoU threshold of 0.5
            matched_ids, _ = match_predictions_to_gt(pred_results, gt_anns, iou_threshold=0.5)
            matched_gt.update(matched_ids)

            # Classify predictions: matched vs false positives
            for pred in pred_results:
                pred_cat_id = pred["category_id"]
                # Check if this prediction matched any GT of the same class
                matched = False
                for gt_ann in gt_anns:
                    if gt_ann["id"] in matched_ids and gt_ann["category_id"] == pred_cat_id:
                        matched = True
                        matched_by_class[pred_cat_id] += 1
                        break
                if not matched:
                    # This prediction didn't match any GT - it's a false positive
                    false_positives += 1
                    fp_by_class[pred_cat_id] += 1

            # Count missed GT objects (false negatives)
            for gt_ann in gt_anns:
                if gt_ann["id"] not in matched_gt:
                    missed_by_class[gt_ann["category_id"]] += 1

        total_gt = len(video_gt_anns)
        false_negatives = total_gt - len(matched_gt)
        mota = 1.0 - (false_negatives + false_positives) / total_gt if total_gt > 0 else 0.0

        # Get category names
        categories = coco_gt.loadCats(coco_gt.getCatIds())
        category_id_to_name = {cat["id"]: cat["name"] for cat in categories}

        per_video_stats[video_name] = {
            "total_gt": total_gt,
            "total_pred": len(video_results),
            "matched": len(matched_gt),
            "false_negatives": false_negatives,
            "false_positives": false_positives,
            "MOTA": mota,
            "precision": (
                len(matched_gt) / (len(matched_gt) + false_positives)
                if (len(matched_gt) + false_positives) > 0
                else 0.0
            ),
            "recall": len(matched_gt) / total_gt if total_gt > 0 else 0.0,
            "matched_by_class": {
                category_id_to_name.get(cid, f"class_{cid}"): count
                for cid, count in matched_by_class.items()
            },
            "missed_by_class": {
                category_id_to_name.get(cid, f"class_{cid}"): count
                for cid, count in missed_by_class.items()
            },
            "fp_by_class": {
                category_id_to_name.get(cid, f"class_{cid}"): count
                for cid, count in fp_by_class.items()
            },
        }

    return per_video_stats


def analyse_mota_per_video(coco_gt, coco_results, test_dataset, frame_idx_to_unique_image_id):
    """Analyse MOTA per video"""
    per_video_stats = compute_mota_per_video(
        coco_gt, coco_results, test_dataset, frame_idx_to_unique_image_id
    )

    print("\n" + "=" * 80)
    print("MOTA ANALYSIS PER VIDEO")
    print("=" * 80)

    for video_name in sorted(per_video_stats.keys()):
        stats = per_video_stats[video_name]
        print(f"\n{video_name}:")
        print(f"  Total GT objects: {stats['total_gt']}")
        print(f"  Total predictions: {stats['total_pred']}")
        print(f"  Matched: {stats['matched']}")
        print(
            f"  False Negatives (missed): {stats['false_negatives']} ({stats['false_negatives']/stats['total_gt']*100:.1f}%)"
        )
        print(
            f"  False Positives: {stats['false_positives']} ({stats['false_positives']/stats['total_gt']*100:.1f}%)"
        )
        print(f"  MOTA: {stats['MOTA']:.4f}")
        print(f"  Precision: {stats['precision']:.4f}")
        print(f"  Recall: {stats['recall']:.4f}")

        print(f"\n  Breakdown by class:")
        all_classes = (
            set(stats["matched_by_class"].keys())
            | set(stats["missed_by_class"].keys())
            | set(stats["fp_by_class"].keys())
        )
        for class_name in sorted(all_classes):
            matched = stats["matched_by_class"].get(class_name, 0)
            missed = stats["missed_by_class"].get(class_name, 0)
            fp = stats["fp_by_class"].get(class_name, 0)
            total_gt_class = matched + missed
            if total_gt_class > 0:
                print(f"    {class_name}:")
                print(f"      GT: {total_gt_class}, Matched: {matched}, Missed: {missed}, FP: {fp}")
                print(
                    f"      Recall: {matched/total_gt_class:.4f}, Precision: {matched/(matched+fp) if (matched+fp) > 0 else 0:.4f}"
                )

    # Compare videos
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)

    video_names = sorted(per_video_stats.keys())
    if len(video_names) >= 2:
        print(f"\n{video_names[0]} vs {video_names[1]}:")
        stats1 = per_video_stats[video_names[0]]
        stats2 = per_video_stats[video_names[1]]

        print(
            f"\n  MOTA: {stats1['MOTA']:.4f} vs {stats2['MOTA']:.4f} (diff: {stats1['MOTA'] - stats2['MOTA']:.4f})"
        )
        print(
            f"  Recall: {stats1['recall']:.4f} vs {stats2['recall']:.4f} (diff: {stats1['recall'] - stats2['recall']:.4f})"
        )
        print(
            f"  Precision: {stats1['precision']:.4f} vs {stats2['precision']:.4f} (diff: {stats1['precision'] - stats2['precision']:.4f})"
        )

        print(
            f"\n  False Negatives: {stats1['false_negatives']} ({stats1['false_negatives']/stats1['total_gt']*100:.1f}%) vs {stats2['false_negatives']} ({stats2['false_negatives']/stats2['total_gt']*100:.1f}%)"
        )
        print(
            f"  False Positives: {stats1['false_positives']} ({stats1['false_positives']/stats1['total_gt']*100:.1f}%) vs {stats2['false_positives']} ({stats2['false_positives']/stats2['total_gt']*100:.1f}%)"
        )

        # Count frames per video
        frame_counts = {}
        for frame_idx, frame_info in enumerate(test_dataset.frame_index):
            video_item = test_dataset.video_items[frame_info["video_idx"]]
            video_path = Path(video_item["video_path"])
            video_name = (
                f"{video_path.parent.parent.name}/{video_path.parent.name}/{video_path.name}"
            )
            frame_counts[video_name] = frame_counts.get(video_name, 0) + 1

        if video_names[0] in frame_counts and video_names[1] in frame_counts:
            print(f"\n  Frames: {frame_counts[video_names[0]]} vs {frame_counts[video_names[1]]}")
            print(
                f"  GT objects per frame: {stats1['total_gt']/frame_counts[video_names[0]]:.2f} vs {stats2['total_gt']/frame_counts[video_names[1]]:.2f}"
            )
            print(
                f"  Predictions per frame: {stats1['total_pred']/frame_counts[video_names[0]]:.2f} vs {stats2['total_pred']/frame_counts[video_names[1]]:.2f}"
            )


def analyse_large_objects(coco_gt, coco_results):
    """
    Debug analysis for large object detection performance.

    Helps identify why APlarge (Average Precision for large objects) might
    be low by analysing:
    - IoU distribution for matched large objects
    - Precision/recall at different IoU thresholds
    - Unmatched large objects (missed GT and false positives)
    - Size analysis (area distributions)

    Args:
        coco_gt: COCO ground truth object
        coco_results: List of prediction dictionaries in COCO results format

    Prints:
        Detailed large object analysis including IoU statistics, precision/recall
        curves, and size distributions.
    """
    # Load detections
    coco_dt = coco_gt.loadRes(coco_results)

    # Get category mapping
    categories = coco_gt.loadCats(coco_gt.getCatIds())
    cat_id_to_name = {cat["id"]: cat["name"] for cat in categories}

    # Filter for large objects only
    large_gt_anns = []
    large_pred_results = []

    for ann in coco_gt.anns.values():
        area = ann["area"]
        if get_area_category(area) == "large":
            large_gt_anns.append(ann)

    for result in coco_results:
        # Estimate area from bbox
        x, y, w, h = result["bbox"]
        area = w * h
        if get_area_category(area) == "large":
            large_pred_results.append(result)

    print(f"\nLarge Object Statistics:")
    print(f"  GT large objects: {len(large_gt_anns)}")
    print(f"  Predicted large objects: {len(large_pred_results)}")

    # Group by image
    large_gt_by_image = defaultdict(list)
    large_pred_by_image = defaultdict(list)

    for ann in large_gt_anns:
        large_gt_by_image[ann["image_id"]].append(ann)

    for result in large_pred_results:
        large_pred_by_image[result["image_id"]].append(result)

    # Analyse IoU distribution
    print("\n" + "=" * 70)
    print("IoU DISTRIBUTION FOR LARGE OBJECTS")
    print("=" * 70)

    # Match predictions to GT
    ious = []
    matched_gt = set()
    matched_pred = set()
    unmatched_gt = []
    unmatched_pred = []

    # For each image, match large objects
    for img_id in sorted(large_gt_by_image.keys() | large_pred_by_image.keys()):
        gt_anns = large_gt_by_image.get(img_id, [])
        pred_results = large_pred_by_image.get(img_id, [])

        # Match predictions to GT
        matched_ids, matched_ious = match_predictions_to_gt(
            pred_results, gt_anns, iou_threshold=0.5
        )
        ious.extend(matched_ious)
        matched_gt.update(matched_ids)

        # Track unmatched predictions
        for pred in pred_results:
            if pred["image_id"] not in [
                ann["image_id"] for ann in gt_anns if ann["id"] in matched_ids
            ]:
                unmatched_pred.append(
                    {
                        "img_id": img_id,
                        "bbox": pred["bbox"],
                        "category": cat_id_to_name.get(pred["category_id"], "unknown"),
                        "score": pred["score"],
                    }
                )

        # Find unmatched GT
        for gt_ann in gt_anns:
            if gt_ann["id"] not in matched_gt:
                unmatched_gt.append(
                    {
                        "img_id": img_id,
                        "bbox": gt_ann["bbox"],
                        "category": cat_id_to_name.get(gt_ann["category_id"], "unknown"),
                        "area": gt_ann["area"],
                    }
                )

    if ious:
        print(f"\nMatched large objects: {len(ious)}")
        print(f"  Mean IoU: {np.mean(ious):.4f}")
        print(f"  Median IoU: {np.median(ious):.4f}")
        print(f"  Min IoU: {np.min(ious):.4f}")
        print(f"  Max IoU: {np.max(ious):.4f}")
        print(f"  Std dev: {np.std(ious):.4f}")

        # IoU distribution
        print(f"\nIoU Distribution:")
        print(
            f"  IoU >= 0.75: {sum(1 for i in ious if i >= 0.75)} ({100*sum(1 for i in ious if i >= 0.75)/len(ious):.1f}%)"
        )
        print(
            f"  IoU >= 0.50: {sum(1 for i in ious if i >= 0.50)} ({100*sum(1 for i in ious if i >= 0.50)/len(ious):.1f}%)"
        )
        print(
            f"  IoU >= 0.30: {sum(1 for i in ious if i >= 0.30)} ({100*sum(1 for i in ious if i >= 0.30)/len(ious):.1f}%)"
        )
        print(
            f"  IoU < 0.30: {sum(1 for i in ious if i < 0.30)} ({100*sum(1 for i in ious if i < 0.30)/len(ious):.1f}%)"
        )

        # Low IoU examples
        low_iou = [i for i in ious if i < 0.5]
        if low_iou:
            print(f"\n  Low IoU matches (< 0.5): {len(low_iou)}")
            print(f"    Mean IoU: {np.mean(low_iou):.4f}")
    else:
        print("\nNo matched large objects found!")

    # Unmatched analysis
    print(f"\nUnmatched large GT objects: {len(unmatched_gt)}")
    if unmatched_gt:
        print(f"  This is {100*len(unmatched_gt)/len(large_gt_anns):.1f}% of all large GT objects")
        # By category
        unmatched_by_cat = defaultdict(int)
        for item in unmatched_gt:
            unmatched_by_cat[item["category"]] += 1
        print(f"  By category:")
        for cat, count in sorted(unmatched_by_cat.items()):
            print(f"    {cat}: {count}")

    print(f"\nUnmatched large predictions: {len(unmatched_pred)}")
    if unmatched_pred:
        print(
            f"  This is {100*len(unmatched_pred)/len(large_pred_results):.1f}% of all large predictions"
        )
        # By category
        unmatched_by_cat = defaultdict(int)
        for item in unmatched_pred:
            unmatched_by_cat[item["category"]] += 1
        print(f"  By category:")
        for cat, count in sorted(unmatched_by_cat.items()):
            print(f"    {cat}: {count}")

    # Precision/Recall at different IoU thresholds
    print("\n" + "=" * 70)
    print("PRECISION/RECALL AT DIFFERENT IoU THRESHOLDS")
    print("=" * 70)

    for iou_thresh in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
        matches_at_thresh = sum(1 for i in ious if i >= iou_thresh)
        precision = matches_at_thresh / len(large_pred_results) if large_pred_results else 0.0
        recall = matches_at_thresh / len(large_gt_anns) if large_gt_anns else 0.0
        print(
            f"IoU >= {iou_thresh:.2f}: Precision={precision:.4f}, Recall={recall:.4f} ({matches_at_thresh} matches)"
        )

    # Size analysis
    print("\n" + "=" * 70)
    print("SIZE ANALYSIS")
    print("=" * 70)

    gt_areas = [ann["area"] for ann in large_gt_anns]
    pred_areas = [r["bbox"][2] * r["bbox"][3] for r in large_pred_results]

    if gt_areas:
        print(f"\nGT large object areas:")
        print(f"  Mean: {np.mean(gt_areas):.0f} pixels²")
        print(f"  Median: {np.median(gt_areas):.0f} pixels²")
        print(f"  Min: {np.min(gt_areas):.0f} pixels²")
        print(f"  Max: {np.max(gt_areas):.0f} pixels²")

    if pred_areas:
        print(f"\nPredicted large object areas:")
        print(f"  Mean: {np.mean(pred_areas):.0f} pixels²")
        print(f"  Median: {np.median(pred_areas):.0f} pixels²")
        print(f"  Min: {np.min(pred_areas):.0f} pixels²")
        print(f"  Max: {np.max(pred_areas):.0f} pixels²")

    # Show examples of low IoU matches
    if ious and len(ious) > 0:
        print("\n" + "=" * 70)
        print("EXAMPLES OF LOW IoU MATCHES (First 10)")
        print("=" * 70)
        print("These are large objects that were detected but with low IoU:")
        print("(This suggests localisation issues - boxes are in the right area but not precise)")

        low_iou_count = sum(1 for i in ious if i < 0.5)
        print(f"\nTotal low IoU matches (< 0.5): {low_iou_count}")
        if low_iou_count > 0:
            low_iou_values = [i for i in ious if i < 0.5]
            print(f"  Mean IoU: {np.mean(low_iou_values):.4f}")
            print(
                f"  This suggests {low_iou_count} large objects were detected but with poor localisation"
            )


def compute_per_video_metrics(coco_gt, coco_results, test_dataset, frame_idx_to_unique_image_id):
    """Compute per-video metrics"""
    video_image_ids = get_video_image_mapping(test_dataset, frame_idx_to_unique_image_id)

    per_video_metrics = {}

    for video_name, img_ids in video_image_ids.items():
        video_gt_anns = []
        for ann in coco_gt.anns.values():
            if ann["image_id"] in img_ids:
                video_gt_anns.append(ann)

        video_results = [r for r in coco_results if r["image_id"] in img_ids]

        if len(video_gt_anns) == 0:
            continue

        video_gt_data = {
            "info": coco_gt.dataset.get("info", {}),
            "licenses": coco_gt.dataset.get("licenses", []),
            "images": [coco_gt.imgs[img_id] for img_id in img_ids if img_id in coco_gt.imgs],
            "annotations": video_gt_anns,
            "categories": coco_gt.loadCats(coco_gt.getCatIds()),
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(video_gt_data, f)
            temp_gt_path = f.name

        try:
            video_coco_gt = COCO(temp_gt_path)
            video_coco_dt = video_coco_gt.loadRes(video_results)

            coco_eval = COCOeval(video_coco_gt, video_coco_dt, "bbox")
            coco_eval.params.imgIds = img_ids
            coco_eval.params.catIds = video_coco_gt.getCatIds()
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

            per_class_ap50 = compute_per_class_ap(video_coco_gt, video_coco_dt, "AP50")
            mota = compute_mota_simple(video_coco_gt, video_results)

            per_video_metrics[video_name] = {
                "AP50-95": coco_eval.stats[0],
                "AP50": coco_eval.stats[1],
                "AP75": coco_eval.stats[2],
                "APlarge": coco_eval.stats[5],
                "per_class_ap50": per_class_ap50,
                "MOTA": mota["MOTA"],
                "gt_count": len(video_gt_anns),
                "pred_count": len(video_results),
            }
        finally:
            if os.path.exists(temp_gt_path):
                os.unlink(temp_gt_path)

    return per_video_metrics


def generate_full_report(output_dir, test_dataset, frame_idx_to_unique_image_id):
    """
    Generate comprehensive evaluation report with all metrics.

    Creates a detailed report comparing three scenarios:
    - All Objects: Complete evaluation on all objects
    - Moving Only: Evaluation on moving objects only
    - Static Only: Evaluation on static objects only

    For each scenario, computes:
    - Overall COCO metrics (AP50-95, AP50, AP75, APsmall, APmedium, APlarge)
    - Per-class AP50
    - MOTA (Multiple Object Tracking Accuracy)
    - Per-video metrics

    Args:
        output_dir: Directory containing GT and results JSON files
        test_dataset: GMIND test dataset object
        frame_idx_to_unique_image_id: Mapping from dataset frame index to COCO image ID

    Note:
        Expects the following files in output_dir:
        - coco_gt.json, coco_results.json (all objects)
        - coco_gt_moving.json, coco_results_moving.json (moving objects)
        - coco_gt_static.json, coco_results_static.json (static objects, optional)

    Prints:
        Comprehensive comparison tables and summary statistics.
    """
    output_dir = Path(output_dir)

    # Define scenarios
    scenarios = [
        ("All Objects", "coco_gt.json", "coco_results.json"),
        ("Moving Only", "coco_gt_moving.json", "coco_results_moving.json"),
        ("Static Only", "coco_gt_static.json", "coco_results_static.json"),
    ]

    print("\n" + "=" * 70)
    print("COMPREHENSIVE EVALUATION REPORT")
    print("=" * 70)

    all_results = []

    for scenario_name, gt_file, results_file in scenarios:
        gt_path = output_dir / gt_file
        results_path = output_dir / results_file

        if not gt_path.exists() or not results_path.exists():
            print(f"\nSkipping {scenario_name}: files not found")
            continue

        print(f"\n{'='*70}")
        print(f"Scenario: {scenario_name}")
        print(f"{'='*70}")

        coco_gt = COCO(str(gt_path))
        with open(results_path, "r") as f:
            coco_results = json.load(f)

        # Overall metrics
        metrics = compute_coco_metrics(coco_gt, coco_results)

        # Per-class AP50
        coco_dt = coco_gt.loadRes(coco_results)
        per_class_ap50 = compute_per_class_ap(coco_gt, coco_dt, "AP50")

        # MOTA
        mota = compute_mota_simple(coco_gt, coco_results)

        # Per-video metrics
        per_video = compute_per_video_metrics(
            coco_gt, coco_results, test_dataset, frame_idx_to_unique_image_id
        )

        all_results.append(
            {
                "scenario": scenario_name,
                "metrics": metrics,
                "per_class_ap50": per_class_ap50,
                "mota": mota,
                "per_video": per_video,
            }
        )

        print(f"\nOverall Metrics:")
        print(f"  AP50-95: {metrics['AP50-95']:.4f}")
        print(f"  AP50: {metrics['AP50']:.4f}")
        print(f"  APlarge: {metrics['APlarge']:.4f}")
        print(f"  MOTA: {mota['MOTA']:.4f}")

        print(f"\nPer-Class AP50:")
        for class_name, ap50 in sorted(per_class_ap50.items()):
            print(f"  {class_name}: {ap50:.4f}")

        print(f"\nPer-Video Metrics:")
        for video_name, video_metrics in sorted(per_video.items()):
            print(f"  {video_name}:")
            print(f"    AP50-95: {video_metrics['AP50-95']:.4f}")
            print(f"    AP50: {video_metrics['AP50']:.4f}")
            print(f"    APlarge: {video_metrics['APlarge']:.4f}")
            print(f"    MOTA: {video_metrics['MOTA']:.4f}")
            print(f"    Per-Class AP50:")
            for class_name, ap50 in sorted(video_metrics["per_class_ap50"].items()):
                print(f"      {class_name}: {ap50:.4f}")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE - AP50 PER CLASS")
    print("=" * 70)
    all_classes = set()
    for result in all_results:
        all_classes.update(result["per_class_ap50"].keys())

    print(f"{'Class':<15} {'All Objects':<15} {'Moving Only':<15} {'Static Only':<15}")
    print("-" * 70)
    for class_name in sorted(all_classes):
        row = f"{class_name:<15}"
        for result in all_results:
            value = result["per_class_ap50"].get(class_name, 0.0)
            row += f"{value:.4f}        "
        print(row)

    print("\n" + "=" * 70)
    print("SUMMARY TABLE - MOTA")
    print("=" * 70)
    print(f"{'Scenario':<20} {'MOTA':<15} {'FN':<15} {'FP':<15}")
    print("-" * 70)
    for result in all_results:
        mota = result["mota"]
        print(
            f"{result['scenario']:<20} {mota['MOTA']:.4f}        {mota['false_negatives']:<15} {mota['false_positives']:<15}"
        )

    print("\n" + "=" * 70)
    print("PER-VIDEO SUMMARY")
    print("=" * 70)

    all_videos = set()
    for result in all_results:
        all_videos.update(result["per_video"].keys())

    for video_name in sorted(all_videos):
        print(f"\n{video_name}:")
        print(f"{'Metric':<15} {'All Objects':<15} {'Moving Only':<15} {'Static Only':<15}")
        print("-" * 70)

        metric_names = ["AP50-95", "AP50", "APlarge", "MOTA"]
        for metric_name in metric_names:
            row = f"{metric_name:<15}"
            for result in all_results:
                video_metrics = result["per_video"].get(video_name, {})
                value = video_metrics.get(metric_name, -1.0)
                if value < 0:
                    row += f"{'N/A':<15}"
                else:
                    row += f"{value:.4f}        "
            print(row)

        # Per-class AP50 per video
        print(f"\n  Per-Class AP50:")
        for class_name in sorted(all_classes):
            row = f"    {class_name:<15}"
            for result in all_results:
                video_metrics = result["per_video"].get(video_name, {})
                per_class = video_metrics.get("per_class_ap50", {})
                value = per_class.get(class_name, 0.0)
                row += f"{value:.4f}        "
            print(row)

    print("\n" + "=" * 70)
    print("Report complete!")
    print("=" * 70)
    print(f"\nSize vs Performance plot: {output_dir / 'size_vs_performance.png'}")


def compute_overall_mota(coco_gt, coco_results):
    """
    Compute overall MOTA (Multiple Object Tracking Accuracy) for the entire dataset.

    This is a detection-based MOTA approximation that computes MOTA across all
    frames, not per-video. For per-video MOTA, use --mota flag.

    Args:
        coco_gt: COCO ground truth object
        coco_results: List of prediction dictionaries in COCO results format

    Prints:
        Overall MOTA statistics including precision, recall, false negatives/positives.
    """
    print("\nComputing overall MOTA (Multiple Object Tracking Accuracy)...")
    print("Note: This is a detection-based MOTA approximation")
    print("Full MOTA requires GT track IDs which aren't available in COCO format")

    # Use the simple MOTA computation from analysis_utils
    mota_results = compute_mota_simple(coco_gt, coco_results)

    print(f"\nOverall MOTA Calculation:")
    print(f"  Total GT objects: {mota_results['total_gt']}")
    print(f"  Matched GT: {mota_results['matched']}")
    print(f"  False Negatives (missed): {mota_results['false_negatives']}")
    print(f"  False Positives: {mota_results['false_positives']}")
    print(f"  ID Switches: N/A (requires tracking information)")
    print(f"\n  MOTA = 1 - (FN + FP) / GT = {mota_results['MOTA']:.4f}")
    print(f"  Precision: {mota_results.get('precision', 0.0):.4f}")
    print(f"  Recall: {mota_results.get('recall', 0.0):.4f}")


def analyse_size_vs_performance(coco_gt, coco_results, output_dir):
    """
    Analyse and plot performance by object size.

    Creates a dual-axis plot showing miss rate and AP50 vs object size,
    helping identify if the model performs differently on different object sizes.

    Args:
        coco_gt: COCO ground truth object
        coco_results: List of prediction dictionaries in COCO results format
        output_dir: Directory to save the plot

    Saves:
        size_vs_performance.png in the output directory
    """
    print("\nAnalysing performance by object size...")

    # Define size bins
    size_bins = [
        (0, 32**2, "tiny"),
        (32**2, 96**2, "small"),
        (96**2, 256**2, "medium"),
        (256**2, 512**2, "large"),
        (512**2, float("inf"), "xlarge"),
    ]

    # Initialise statistics containers
    bin_stats = {
        bin_name: {"gt_objects": [], "matched": [], "missed": [], "areas": []}
        for _, _, bin_name in size_bins
    }

    # Group GT and predictions by image
    gt_by_image, pred_by_image = group_by_image(coco_gt, coco_results)

    # Match predictions to GT and categorise by size
    for img_id in sorted(coco_gt.imgs.keys()):
        gt_anns = gt_by_image.get(img_id, [])
        pred_results = pred_by_image.get(img_id, [])

        matched_gt_ids = set()

        # Match predictions to GT
        for pred in pred_results:
            pred_bbox = pred["bbox"]
            pred_cat_id = pred["category_id"]

            best_iou = 0.0
            best_gt_id = None

            for gt_ann in gt_anns:
                if gt_ann["category_id"] != pred_cat_id:
                    continue

                iou = compute_iou(pred_bbox, gt_ann["bbox"])
                if iou > best_iou and iou >= 0.5:
                    best_iou = iou
                    best_gt_id = gt_ann["id"]

            if best_gt_id is not None:
                matched_gt_ids.add(best_gt_id)

        # Categorise GT objects by size
        for gt_ann in gt_anns:
            area = gt_ann.get("area", gt_ann["bbox"][2] * gt_ann["bbox"][3])

            # Find appropriate bin
            bin_name = None
            for min_area, max_area, name in size_bins:
                if min_area <= area < max_area:
                    bin_name = name
                    break

            if bin_name:
                bin_stats[bin_name]["gt_objects"].append(gt_ann)
                bin_stats[bin_name]["areas"].append(area)

                if gt_ann["id"] in matched_gt_ids:
                    bin_stats[bin_name]["matched"].append(gt_ann)
                else:
                    bin_stats[bin_name]["missed"].append(gt_ann)

    # Compute metrics per bin
    bin_metrics = {}
    for bin_name, stats in bin_stats.items():
        total = len(stats["gt_objects"])
        matched = len(stats["matched"])
        missed = len(stats["missed"])

        if total == 0:
            continue

        miss_rate = missed / total

        # Compute AP50 for this bin
        ap50 = 0.0
        if matched > 0:
            bin_gt_anns = stats["gt_objects"]
            bin_matched_ids = {ann["id"] for ann in stats["matched"]}

            # Get predictions that match these GT objects
            bin_results = []
            for img_id in sorted(coco_gt.imgs.keys()):
                gt_anns = gt_by_image.get(img_id, [])
                pred_results = pred_by_image.get(img_id, [])

                for pred in pred_results:
                    for gt_ann in gt_anns:
                        if gt_ann["id"] in bin_matched_ids:
                            if gt_ann["category_id"] == pred["category_id"]:
                                iou = compute_iou(pred["bbox"], gt_ann["bbox"])
                                if iou >= 0.5:
                                    bin_results.append(pred)
                                    break

            if len(bin_gt_anns) > 0 and len(bin_results) > 0:
                bin_gt_data = {
                    "info": coco_gt.dataset.get("info", {}),
                    "licenses": coco_gt.dataset.get("licenses", []),
                    "images": [coco_gt.imgs[ann["image_id"]] for ann in bin_gt_anns],
                    "annotations": bin_gt_anns,
                    "categories": coco_gt.loadCats(coco_gt.getCatIds()),
                }

                with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                    json.dump(bin_gt_data, f)
                    temp_gt_path = f.name

                try:
                    bin_coco_gt = COCO(temp_gt_path)
                    bin_coco_dt = bin_coco_gt.loadRes(bin_results)

                    bin_coco_eval = COCOeval(bin_coco_gt, bin_coco_dt, "bbox")
                    bin_coco_eval.params.imgIds = list(bin_coco_gt.imgs.keys())
                    bin_coco_eval.params.catIds = bin_coco_gt.getCatIds()
                    bin_coco_eval.evaluate()
                    bin_coco_eval.accumulate()
                    bin_coco_eval.summarize()

                    ap50 = bin_coco_eval.stats[1]  # AP50
                finally:
                    if os.path.exists(temp_gt_path):
                        os.unlink(temp_gt_path)

        mean_area = np.mean(stats["areas"]) if stats["areas"] else 0.0

        bin_metrics[bin_name] = {
            "total": total,
            "matched": matched,
            "missed": missed,
            "miss_rate": miss_rate,
            "ap50": ap50,
            "mean_area": mean_area,
        }

    # Print summary
    print("\n" + "=" * 70)
    print("PERFORMANCE BY OBJECT SIZE")
    print("=" * 70)
    print(
        f"{'Size Bin':<15} {'Total':<10} {'Matched':<10} {'Missed':<10} {'Miss Rate':<12} {'AP50':<10} {'Mean Area':<12}"
    )
    print("-" * 90)
    for bin_name in ["tiny", "small", "medium", "large", "xlarge"]:
        if bin_name in bin_metrics:
            m = bin_metrics[bin_name]
            print(
                f"{bin_name:<15} {m['total']:<10} {m['matched']:<10} {m['missed']:<10} {m['miss_rate']:.4f}        {m['ap50']:.4f}        {m['mean_area']:.0f}        "
            )

    # Create plot
    sorted_bins = sorted(bin_metrics.items(), key=lambda x: x[1]["mean_area"])
    bin_names = [name for name, _ in sorted_bins]
    mean_areas = [metrics["mean_area"] for _, metrics in sorted_bins]
    miss_rates = [metrics["miss_rate"] for _, metrics in sorted_bins]
    ap50s = [metrics["ap50"] for _, metrics in sorted_bins]

    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Primary y-axis: Miss rate
    color1 = "tab:red"
    ax1.set_xlabel("Object Size (Area in pixels²)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Miss Rate", fontsize=12, fontweight="bold")
    line1 = ax1.plot(
        mean_areas,
        miss_rates,
        "o-",
        color=color1,
        linewidth=2.5,
        markersize=8,
        label="Miss Rate",
        linestyle="-",
        marker="o",
        markeredgewidth=1.5,
        markeredgecolor="black",
    )
    ax1.tick_params(axis="y")
    ax1.set_ylim([0, max(miss_rates) * 1.1 if miss_rates else 1.0])
    ax1.grid(True, alpha=0.3)

    from matplotlib.ticker import ScalarFormatter

    ax1.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax1.ticklabel_format(style="scientific", axis="x", scilimits=(0, 0))
    ax1.xaxis.offsetText.set_fontsize(10)

    # Secondary y-axis: AP50
    ax2 = ax1.twinx()
    color2 = "tab:blue"
    ax2.set_ylabel("AP50", fontsize=12, fontweight="bold")
    line2 = ax2.plot(
        mean_areas,
        ap50s,
        "s--",
        color=color2,
        linewidth=2.5,
        markersize=8,
        label="AP50",
        linestyle="--",
        marker="s",
        markeredgewidth=1.5,
        markeredgecolor="black",
    )
    ax2.tick_params(axis="y")
    ax2.set_ylim([0, max(ap50s) * 1.1 if ap50s else 1.0])

    # Add bin labels
    for i, (name, area) in enumerate(zip(bin_names, mean_areas)):
        ax1.annotate(
            name,
            (area, miss_rates[i]),
            textcoords="offset points",
            xytext=(15, 0),
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.7),
        )

    plt.title("Object Size vs Miss Rate and AP50", fontsize=14, fontweight="bold", pad=20)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left")

    plt.tight_layout()
    output_path = Path(output_dir) / "size_vs_performance.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved plot to: {output_path}")


def compare_static_vs_moving(output_dir, test_dataset, frame_idx_to_unique_image_id):
    """Compare metrics for static vs moving objects"""
    output_dir = Path(output_dir)

    # Define scenarios
    scenarios = [
        {
            "name": "All Objects",
            "gt": output_dir / "coco_gt.json",
            "results": output_dir / "coco_results.json",
        },
        {
            "name": "Moving Objects Only",
            "gt": output_dir / "coco_gt_moving.json",
            "results": output_dir / "coco_results_moving.json",
        },
    ]

    # Check if static objects file exists, if not create it
    static_gt = output_dir / "coco_gt_static.json"
    static_results = output_dir / "coco_results_static.json"

    if static_gt.exists() and static_results.exists():
        scenarios.append(
            {"name": "Static Objects Only", "gt": static_gt, "results": static_results}
        )
    else:
        print("\nStatic objects files not found. Creating them...")
        # Load all GT and moving GT to compute static
        all_gt = COCO(str(output_dir / "coco_gt.json"))
        moving_gt = COCO(str(output_dir / "coco_gt_moving.json"))

        moving_ann_ids = set(moving_gt.anns.keys())
        static_ann_ids = set(all_gt.anns.keys()) - moving_ann_ids

        print(f"Found {len(static_ann_ids)} static objects (out of {len(all_gt.anns)})")

        # Create static GT
        new_images = []
        new_annotations = []
        new_categories = all_gt.loadCats(all_gt.getCatIds())

        for img_id, img_info in all_gt.imgs.items():
            new_images.append(
                {
                    "id": img_id,
                    "width": img_info["width"],
                    "height": img_info["height"],
                    "file_name": img_info.get("file_name", f"frame_{img_id:06d}.jpg"),
                }
            )

        ann_id = 1
        for ann_id_old, ann in all_gt.anns.items():
            if ann_id_old in static_ann_ids:
                new_ann = ann.copy()
                new_ann["id"] = ann_id
                new_annotations.append(new_ann)
                ann_id += 1

        static_coco_data = {
            "info": all_gt.dataset.get("info", {}),
            "licenses": all_gt.dataset.get("licenses", []),
            "images": new_images,
            "annotations": new_annotations,
            "categories": new_categories,
        }

        with open(static_gt, "w") as f:
            json.dump(static_coco_data, f, indent=2)

        # Filter results to match static GT
        with open(output_dir / "coco_results.json", "r") as f:
            all_results = json.load(f)

        # Match predictions to static GT
        static_coco_gt = COCO(str(static_gt))
        static_results_list = []

        gt_by_image, pred_by_image = group_by_image(static_coco_gt, all_results)

        for img_id in sorted(pred_by_image.keys()):
            pred_results = pred_by_image[img_id]
            static_gt_anns = gt_by_image.get(img_id, [])

            matched_ids, _ = match_predictions_to_gt(
                pred_results, static_gt_anns, iou_threshold=0.5
            )
            # Add matched predictions
            for pred in pred_results:
                for gt_ann in static_gt_anns:
                    if gt_ann["id"] in matched_ids:
                        static_results_list.append(pred)
                        break

        with open(static_results, "w") as f:
            json.dump(static_results_list, f, indent=2)

        print(f"Created static GT: {len(new_annotations)} annotations")
        print(f"Created static results: {len(static_results_list)} detections")

        scenarios.append(
            {"name": "Static Objects Only", "gt": static_gt, "results": static_results}
        )

    # Evaluate all scenarios
    all_results = []
    for scenario in scenarios:
        if scenario["gt"].exists() and scenario["results"].exists():
            coco_gt = COCO(str(scenario["gt"]))
            with open(scenario["results"], "r") as f:
                coco_results = json.load(f)

            print(f"\n{'='*70}")
            print(f"Evaluating: {scenario['name']}")
            print(f"{'='*70}")

            print(f"GT: {len(coco_gt.imgs)} images, {len(coco_gt.anns)} annotations")
            print(f"Results: {len(coco_results)} detections")

            # Compute overall metrics
            metrics = compute_coco_metrics(coco_gt, coco_results)

            # Compute per-class AP (both AP50-95 and AP50)
            coco_dt = coco_gt.loadRes(coco_results)
            per_class_ap50_95 = compute_per_class_ap(coco_gt, coco_dt, "AP50-95")
            per_class_ap50 = compute_per_class_ap(coco_gt, coco_dt, "AP50")

            # Compute per-video metrics if dataset is provided
            per_video_metrics = {}
            if test_dataset is not None and frame_idx_to_unique_image_id is not None:
                per_video_metrics = compute_per_video_metrics(
                    coco_gt, coco_results, test_dataset, frame_idx_to_unique_image_id
                )

            all_results.append(
                {
                    "scenario": scenario["name"],
                    "gt_count": len(coco_gt.anns),
                    "pred_count": len(coco_results),
                    "metrics": metrics,
                    "per_class_ap50_95": per_class_ap50_95,
                    "per_class_ap50": per_class_ap50,
                    "per_video_metrics": per_video_metrics,
                }
            )
        else:
            print(f"\nWarning: Files not found for {scenario['name']}")
            print(f"  GT: {scenario['gt']}")
            print(f"  Results: {scenario['results']}")

    # Print comprehensive comparison
    print("\n" + "=" * 70)
    print("COMPREHENSIVE RESULTS BREAKDOWN")
    print("=" * 70)

    # Overall metrics comparison
    print("\n" + "=" * 70)
    print("OVERALL METRICS COMPARISON")
    print("=" * 70)
    print(f"{'Metric':<15} {'All Objects':<15} {'Moving Only':<15} {'Static Only':<15}")
    print("-" * 70)

    metric_names = ["AP50-95", "AP50", "AP75", "APsmall", "APmedium", "APlarge"]
    for metric_name in metric_names:
        row = f"{metric_name:<15}"
        for result in all_results:
            value = result["metrics"].get(metric_name, -1.0)
            if value < 0:
                row += f"{'N/A':<15}"
            else:
                row += f"{value:.4f}        "
        print(row)

    # Per-class comparison - AP50-95
    print("\n" + "=" * 70)
    print("PER-CLASS AP50-95 COMPARISON")
    print("=" * 70)

    # Get all class names
    all_classes = set()
    for result in all_results:
        all_classes.update(result.get("per_class_ap50_95", {}).keys())

    print(f"{'Class':<15} {'All Objects':<15} {'Moving Only':<15} {'Static Only':<15}")
    print("-" * 70)
    for class_name in sorted(all_classes):
        row = f"{class_name:<15}"
        for result in all_results:
            value = result.get("per_class_ap50_95", {}).get(class_name, 0.0)
            row += f"{value:.4f}        "
        print(row)

    # Per-class comparison - AP50
    print("\n" + "=" * 70)
    print("PER-CLASS AP50 COMPARISON")
    print("=" * 70)

    print(f"{'Class':<15} {'All Objects':<15} {'Moving Only':<15} {'Static Only':<15}")
    print("-" * 70)
    for class_name in sorted(all_classes):
        row = f"{class_name:<15}"
        for result in all_results:
            value = result.get("per_class_ap50", {}).get(class_name, 0.0)
            row += f"{value:.4f}        "
        print(row)

    # Counts comparison
    print("\n" + "=" * 70)
    print("OBJECT COUNTS")
    print("=" * 70)
    print(f"{'Scenario':<20} {'GT Objects':<15} {'Predictions':<15} {'Ratio':<15}")
    print("-" * 70)
    for result in all_results:
        ratio = result["pred_count"] / result["gt_count"] if result["gt_count"] > 0 else 0.0
        print(
            f"{result['scenario']:<20} {result['gt_count']:<15} {result['pred_count']:<15} {ratio:.4f}        "
        )

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    if len(all_results) >= 2:
        all_obj = all_results[0]
        moving_obj = all_results[1]

        print(f"\nMoving vs All Objects:")
        print(
            f"  GT objects: {moving_obj['gt_count']} / {all_obj['gt_count']} ({100*moving_obj['gt_count']/all_obj['gt_count']:.1f}%)"
        )
        print(
            f"  Predictions: {moving_obj['pred_count']} / {all_obj['pred_count']} ({100*moving_obj['pred_count']/all_obj['pred_count']:.1f}%)"
        )
        print(
            f"  AP50-95: {moving_obj['metrics']['AP50-95']:.4f} vs {all_obj['metrics']['AP50-95']:.4f} ({100*(moving_obj['metrics']['AP50-95']/all_obj['metrics']['AP50-95']-1):+.1f}%)"
        )
        print(
            f"  APlarge: {moving_obj['metrics']['APlarge']:.4f} vs {all_obj['metrics']['APlarge']:.4f} ({100*(moving_obj['metrics']['APlarge']/all_obj['metrics']['APlarge']-1):+.1f}%)"
        )

    if len(all_results) >= 3:
        static_obj = all_results[2]
        print(f"\nStatic vs All Objects:")
        print(
            f"  GT objects: {static_obj['gt_count']} / {all_obj['gt_count']} ({100*static_obj['gt_count']/all_obj['gt_count']:.1f}%)"
        )
        print(
            f"  Predictions: {static_obj['pred_count']} / {all_obj['pred_count']} ({100*static_obj['pred_count']/all_obj['pred_count']:.1f}%)"
        )
        print(
            f"  AP50-95: {static_obj['metrics']['AP50-95']:.4f} vs {all_obj['metrics']['AP50-95']:.4f} ({100*(static_obj['metrics']['AP50-95']/all_obj['metrics']['AP50-95']-1):+.1f}%)"
        )
        print(
            f"  APlarge: {static_obj['metrics']['APlarge']:.4f} vs {all_obj['metrics']['APlarge']:.4f} ({100*(static_obj['metrics']['APlarge']/all_obj['metrics']['APlarge']-1):+.1f}%)"
        )

    # Per-video comparison
    if all_results and all_results[0].get("per_video_metrics"):
        print("\n" + "=" * 70)
        print("PER-VIDEO METRICS COMPARISON")
        print("=" * 70)

        # Get all video names
        all_videos = set()
        for result in all_results:
            all_videos.update(result.get("per_video_metrics", {}).keys())

        for video_name in sorted(all_videos):
            print(f"\n{video_name}:")
            print(f"{'Metric':<15} {'All Objects':<15} {'Moving Only':<15} {'Static Only':<15}")
            print("-" * 70)

            metric_names = ["AP50-95", "AP50", "AP75", "APlarge"]
            for metric_name in metric_names:
                row = f"{metric_name:<15}"
                for result in all_results:
                    video_metrics = result.get("per_video_metrics", {}).get(video_name, {})
                    value = video_metrics.get(metric_name, -1.0)
                    if value < 0:
                        row += f"{'N/A':<15}"
                    else:
                        row += f"{value:.4f}        "
                print(row)

            # Object counts
            print(f"\n  Object Counts:")
            for result in all_results:
                video_metrics = result.get("per_video_metrics", {}).get(video_name, {})
                gt_count = video_metrics.get("gt_count", 0)
                pred_count = video_metrics.get("pred_count", 0)
                ratio = pred_count / gt_count if gt_count > 0 else 0.0
                print(
                    f"    {result['scenario']:<20}: GT={gt_count}, Pred={pred_count}, Ratio={ratio:.3f}"
                )

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)


# ============================================================================
# Main Function
# ============================================================================


def load_data(args):
    """Load GT and results data"""
    # Load results
    results_path = Path(args.results) if args.results else None
    if results_path and not results_path.exists():
        # Try relative to output dir
        alt_path = Path(args.output_dir) / results_path.name
        if alt_path.exists():
            results_path = alt_path
        elif not results_path.is_absolute():
            results_path = Path(args.output_dir) / results_path

    if results_path:
        print(f"Loading results from: {results_path}")
        with open(results_path, "r") as f:
            coco_results = json.load(f)
    else:
        # Try default location
        default_results = Path(args.output_dir) / "coco_results.json"
        if default_results.exists():
            print(f"Loading results from: {default_results}")
            with open(default_results, "r") as f:
                coco_results = json.load(f)
        else:
            raise FileNotFoundError(
                f"Results file not found. Specify --results or ensure {default_results} exists"
            )

    # Find or create GT file
    if args.gt:
        gt_path = Path(args.gt)
        if not gt_path.is_absolute():
            gt_path = Path(args.output_dir) / gt_path
        print(f"Loading GT from: {gt_path}")
        coco_gt = COCO(str(gt_path))
        categories = coco_gt.loadCats(coco_gt.getCatIds())
        category_name_to_id = {cat["name"]: cat["id"] for cat in categories}
    else:
        # Try default location
        default_gt = Path(args.output_dir) / "coco_gt.json"
        if default_gt.exists():
            print(f"Loading GT from: {default_gt}")
            coco_gt = COCO(str(default_gt))
            categories = coco_gt.loadCats(coco_gt.getCatIds())
            category_name_to_id = {cat["name"]: cat["id"] for cat in categories}
        else:
            # Create GT from dataset
            print("Creating GT from dataset...")
            config_path = Path(__file__).parent.parent.parent / "DeepLearning" / "gmind_config.yaml"
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

            coco_gt, temp_gt_path, category_name_to_id, _ = create_coco_gt_from_dataset(
                test_dataset
            )
            print(f"Created GT: {len(coco_gt.imgs)} images, {len(coco_gt.anns)} annotations")

    return coco_gt, coco_results, category_name_to_id


def load_dataset():
    """Load test dataset for per-video analysis"""
    config_path = Path(__file__).parent.parent.parent / "DeepLearning" / "gmind_config.yaml"
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

    frame_idx_to_unique_image_id = {}
    for frame_idx in range(len(test_dataset)):
        frame_idx_to_unique_image_id[frame_idx] = frame_idx + 1

    return test_dataset, frame_idx_to_unique_image_id


def main():
    parser = argparse.ArgumentParser(
        description="Unified analysis script for evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all analyses
  python -m Evaluation.analysis.analyse_results --all

  # Run specific analyses
  python -m Evaluation.analysis.analyse_results --detection --missed

  # Generate full report
  python -m Evaluation.analysis.analyse_results --full-report

  # Compare static vs moving
  python -m Evaluation.analysis.analyse_results --compare-static-moving
        """,
    )

    # Input/output arguments
    parser.add_argument(
        "--results",
        type=str,
        help="Path to COCO results JSON file (default: output_dir/coco_results.json)",
    )
    parser.add_argument(
        "--gt",
        type=str,
        help="Path to COCO GT JSON file (default: output_dir/coco_gt.json or create from dataset)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="baseline_evaluation_results",
        help="Output directory (default: baseline_evaluation_results)",
    )

    # Analysis selection flags
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all analyses (equivalent to --detection --missed --mota --large --full-report --compare-static-moving)",
    )
    parser.add_argument(
        "--detection",
        action="store_true",
        help="Analyse detection results (objects per frame, class breakdown, GT overlaps)",
    )
    parser.add_argument(
        "--missed",
        action="store_true",
        help="Analyse missed objects per frame (especially large objects)",
    )
    parser.add_argument(
        "--mota",
        action="store_true",
        help="Analyse MOTA per video with detailed breakdown",
    )
    parser.add_argument(
        "--overall-mota",
        action="store_true",
        help="Compute overall MOTA for the entire dataset",
    )
    parser.add_argument(
        "--size-plot",
        action="store_true",
        help="Plot object size vs miss rate and AP50 (saves size_vs_performance.png)",
    )
    parser.add_argument(
        "--large",
        action="store_true",
        help="Debug why APlarge is low (IoU distribution, precision/recall for large objects)",
    )
    parser.add_argument(
        "--full-report",
        action="store_true",
        help="Generate comprehensive evaluation report (AP50, MOTA, per-video metrics)",
    )
    parser.add_argument(
        "--compare-static-moving",
        action="store_true",
        help="Compare metrics for static vs moving objects",
    )

    args = parser.parse_args()

    # If --all is specified, enable all analyses
    if args.all:
        args.detection = True
        args.missed = True
        args.mota = True
        args.overall_mota = True
        args.size_plot = True
        args.large = True
        args.full_report = True
        args.compare_static_moving = True

    # If no analysis is selected, show help
    if not any(
        [
            args.detection,
            args.missed,
            args.mota,
            args.overall_mota,
            args.size_plot,
            args.large,
            args.full_report,
            args.compare_static_moving,
        ]
    ):
        parser.print_help()
        print("\nError: No analysis selected. Use --all or specify individual analyses.")
        sys.exit(1)

    # Load data
    try:
        coco_gt, coco_results, category_name_to_id = load_data(args)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Load dataset if needed for per-video analyses
    test_dataset = None
    frame_idx_to_unique_image_id = None
    if args.mota or args.full_report or args.compare_static_moving:
        print("\nLoading dataset for per-video analysis...")
        test_dataset, frame_idx_to_unique_image_id = load_dataset()

    # Run selected analyses
    if args.detection:
        print("\n" + "=" * 70)
        print("DETECTION RESULTS ANALYSIS")
        print("=" * 70)
        analyse_detection_results(coco_gt, coco_results, category_name_to_id)

    if args.missed:
        print("\n" + "=" * 70)
        print("MISSED OBJECTS PER FRAME ANALYSIS")
        print("=" * 70)
        analyse_missed_per_frame(coco_gt, coco_results)

    if args.mota:
        print("\n" + "=" * 70)
        print("MOTA PER VIDEO ANALYSIS")
        print("=" * 70)
        analyse_mota_per_video(coco_gt, coco_results, test_dataset, frame_idx_to_unique_image_id)

    if args.overall_mota:
        print("\n" + "=" * 70)
        print("OVERALL MOTA ANALYSIS")
        print("=" * 70)
        compute_overall_mota(coco_gt, coco_results)

    if args.size_plot:
        print("\n" + "=" * 70)
        print("SIZE VS PERFORMANCE ANALYSIS")
        print("=" * 70)
        analyse_size_vs_performance(coco_gt, coco_results, args.output_dir)

    if args.large:
        print("\n" + "=" * 70)
        print("LARGE OBJECT ANALYSIS")
        print("=" * 70)
        analyse_large_objects(coco_gt, coco_results)

    if args.full_report:
        print("\n" + "=" * 70)
        print("FULL REPORT GENERATION")
        print("=" * 70)
        generate_full_report(args.output_dir, test_dataset, frame_idx_to_unique_image_id)

    if args.compare_static_moving:
        print("\n" + "=" * 70)
        print("STATIC VS MOVING COMPARISON")
        print("=" * 70)
        compare_static_vs_moving(args.output_dir, test_dataset, frame_idx_to_unique_image_id)

    print("\n" + "=" * 70)
    print("All analyses complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
