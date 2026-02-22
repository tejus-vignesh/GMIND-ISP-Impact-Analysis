"""
Common utility functions for evaluation analysis scripts.

This module provides reusable functions for:
- Computing Intersection over Union (IoU) between bounding boxes
- Categorising objects by size (small/medium/large)
- Matching predictions to ground truth annotations
- Computing COCO evaluation metrics
- Grouping and organising annotation data
- Distance-binned evaluation metrics

All functions use COCO format conventions:
- Bounding boxes: [x, y, width, height] where (x, y) is top-left corner
- Area categories: small (< 32²), medium (32² - 96²), large (≥ 96²)
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1: First bounding box in COCO format [x, y, width, height]
        box2: Second bounding box in COCO format [x, y, width, height]

    Returns:
        IoU value between 0.0 and 1.0. Returns 0.0 if boxes don't overlap
        or if either box has zero area.

    Example:
        >>> box1 = [10, 10, 20, 20]  # x=10, y=10, w=20, h=20
        >>> box2 = [15, 15, 20, 20]  # x=15, y=15, w=20, h=20
        >>> iou = compute_iou(box1, box2)
        >>> print(f"IoU: {iou:.3f}")
    """
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


def get_area_category(area: float) -> str:
    """
    Categorise object area according to COCO evaluation standards.

    Args:
        area: Object area in pixels² (typically width × height)

    Returns:
        Size category string: "small", "medium", or "large"
        - small: area < 32² = 1,024 pixels²
        - medium: 32² ≤ area < 96² = 9,216 pixels²
        - large: area ≥ 96² = 9,216 pixels²

    Note:
        These thresholds match the COCO evaluation protocol used in
        pycocotools for computing APsmall, APmedium, and APlarge metrics.
    """
    if area < 32**2:
        return "small"
    elif area < 96**2:
        return "medium"
    else:
        return "large"


def group_by_image(
    coco_gt: COCO, coco_results: List[Dict]
) -> Tuple[Dict[int, List[Dict]], Dict[int, List[Dict]]]:
    """
    Group ground truth annotations and predictions by image ID.

    This function organises annotations and predictions into per-image
    collections, which is useful for frame-by-frame analysis.

    Args:
        coco_gt: COCO ground truth object containing annotations
        coco_results: List of prediction dictionaries in COCO results format.
                     Each dict must have an "image_id" key.

    Returns:
        Tuple of (gt_by_image, pred_by_image):
        - gt_by_image: Dict mapping image_id to list of GT annotation dicts
        - pred_by_image: Dict mapping image_id to list of prediction dicts

    Example:
        >>> gt_by_img, pred_by_img = group_by_image(coco_gt, coco_results)
        >>> frame_1_gt = gt_by_img[1]  # Get all GT annotations for image 1
        >>> frame_1_pred = pred_by_img[1]  # Get all predictions for image 1
    """
    gt_by_image = defaultdict(list)
    pred_by_image = defaultdict(list)

    for ann in coco_gt.anns.values():
        gt_by_image[ann["image_id"]].append(ann)

    for result in coco_results:
        pred_by_image[result["image_id"]].append(result)

    return gt_by_image, pred_by_image


def match_predictions_to_gt(
    pred_results: List[Dict],
    gt_anns: List[Dict],
    iou_threshold: float = 0.5,
    require_same_class: bool = True,
) -> Tuple[set, List[float]]:
    """
    Match predictions to ground truth annotations using greedy IoU matching.

    For each prediction, finds the best matching GT annotation based on IoU.
    Uses a greedy matching strategy: each prediction is matched to the GT
    with highest IoU that meets the threshold and class requirements.

    Args:
        pred_results: List of prediction dictionaries. Each must have:
                     - "bbox": [x, y, w, h] bounding box
                     - "category_id": integer class ID
        gt_anns: List of ground truth annotation dictionaries. Each must have:
                 - "bbox": [x, y, w, h] bounding box
                 - "category_id": integer class ID
                 - "id": unique annotation ID
        iou_threshold: Minimum IoU required for a match (default: 0.5)
        require_same_class: If True, only match predictions to GT of same class
                           (default: True)

    Returns:
        Tuple of (matched_gt_ids, matched_ious):
        - matched_gt_ids: Set of GT annotation IDs that were matched
        - matched_ious: List of IoU values for each match (same order as matches)

    Note:
        This is a greedy matching algorithm. For optimal matching (e.g., using
        Hungarian algorithm), consider using pycocotools' evaluation functions.
    """
    matched_gt_ids = set()
    matched_ious = []

    for pred in pred_results:
        pred_bbox = pred["bbox"]
        pred_cat_id = pred["category_id"]

        best_iou = 0.0
        best_gt_id = None

        for gt_ann in gt_anns:
            if require_same_class and gt_ann["category_id"] != pred_cat_id:
                continue

            iou = compute_iou(pred_bbox, gt_ann["bbox"])
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_gt_id = gt_ann["id"]

        if best_gt_id is not None:
            matched_gt_ids.add(best_gt_id)
            matched_ious.append(best_iou)

    return matched_gt_ids, matched_ious


def compute_coco_metrics(coco_gt: COCO, coco_results: List[Dict]) -> Dict[str, float]:
    """
    Compute standard COCO evaluation metrics.

    Computes Average Precision (AP) metrics at various IoU thresholds and
    for different object sizes, following the COCO evaluation protocol.

    Args:
        coco_gt: COCO ground truth object
        coco_results: List of prediction dictionaries in COCO results format.
                     Each dict must have: "image_id", "category_id", "bbox", "score"

    Returns:
        Dictionary containing the following metrics:
        - "AP50-95": Average Precision over IoU thresholds 0.5:0.05:0.95
        - "AP50": Average Precision at IoU threshold 0.5
        - "AP75": Average Precision at IoU threshold 0.75
        - "APsmall": AP for small objects (area < 32²)
        - "APmedium": AP for medium objects (32² ≤ area < 96²)
        - "APlarge": AP for large objects (area ≥ 96²)
        - "AR1": Average Recall with max 1 detection per image
        - "AR10": Average Recall with max 10 detections per image
        - "AR100": Average Recall with max 100 detections per image

    Note:
        This function uses pycocotools' COCOeval which implements the official
        COCO evaluation protocol. The evaluation is performed on all images
        and categories in the provided COCO objects.
    """
    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.imgIds = list(coco_gt.imgs.keys())
    coco_eval.params.catIds = coco_gt.getCatIds()
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return {
        "AP50-95": coco_eval.stats[0],
        "AP50": coco_eval.stats[1],
        "AP75": coco_eval.stats[2],
        "APsmall": coco_eval.stats[3],
        "APmedium": coco_eval.stats[4],
        "APlarge": coco_eval.stats[5],
        "AR1": coco_eval.stats[6],
        "AR10": coco_eval.stats[7],
        "AR100": coco_eval.stats[8],
    }


def compute_per_class_ap(
    coco_gt: COCO,
    coco_dt: COCO,
    metric: str = "AP50-95",
    img_ids: Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    Compute Average Precision (AP) for each object class separately.

    Evaluates each class independently, which is useful for understanding
    which classes perform better or worse than others.

    Args:
        coco_gt: COCO ground truth object
        coco_dt: COCO detections object (created via coco_gt.loadRes())
        metric: Which AP metric to compute:
                - "AP50-95": Average Precision over IoU 0.5:0.05:0.95 (default)
                - "AP50": Average Precision at IoU threshold 0.5
                - "AP75": Average Precision at IoU threshold 0.75
        img_ids: Optional list of image IDs to evaluate on. If None, all
                 images in coco_gt are used.

    Returns:
        Dictionary mapping class name (string) to AP value (float).
        Keys are category names from the COCO dataset.

    Example:
        >>> coco_dt = coco_gt.loadRes(coco_results)
        >>> per_class_ap = compute_per_class_ap(coco_gt, coco_dt, "AP50")
        >>> print(f"Car AP50: {per_class_ap['car']:.3f}")
        >>> print(f"Person AP50: {per_class_ap['person']:.3f}")
    """
    categories = coco_gt.loadCats(coco_gt.getCatIds())
    per_class_ap = {}

    stat_idx_map = {"AP50-95": 0, "AP50": 1, "AP75": 2}
    stat_idx = stat_idx_map.get(metric, 0)

    eval_img_ids = img_ids if img_ids is not None else list(coco_gt.imgs.keys())

    for cat in categories:
        cat_id = cat["id"]
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.params.imgIds = eval_img_ids
        coco_eval.params.catIds = [cat_id]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        per_class_ap[cat["name"]] = coco_eval.stats[stat_idx]

    return per_class_ap


def compute_mota_simple(coco_gt: COCO, coco_results: List[Dict]) -> Dict[str, float]:
    """
    Compute a simplified Multiple Object Tracking Accuracy (MOTA) metric.

    This is a detection-based MOTA approximation that doesn't require
    tracking information. It computes:
        MOTA = 1 - (FN + FP) / GT

    Note: Full MOTA includes ID switches (IDSW), but this requires
    tracking information which isn't available in standard COCO format.

    Args:
        coco_gt: COCO ground truth object
        coco_results: List of prediction dictionaries in COCO results format

    Returns:
        Dictionary containing:
        - "MOTA": MOTA score (float, typically between 0.0 and 1.0)
        - "false_negatives": Number of missed GT objects (int)
        - "false_positives": Number of false positive detections (int)
        - "matched": Number of correctly matched GT objects (int)
        - "total_gt": Total number of GT objects (int)

    Note:
        This uses IoU threshold of 0.5 for matching, which is standard
        for detection evaluation. For tracking evaluation, lower thresholds
        (e.g., 0.3) are sometimes used.
    """
    gt_by_image, pred_by_image = group_by_image(coco_gt, coco_results)

    total_gt = len(coco_gt.anns)
    matched_gt = set()
    false_positives = 0

    for img_id in sorted(coco_gt.imgs.keys()):
        gt_anns = gt_by_image.get(img_id, [])
        pred_results = pred_by_image.get(img_id, [])

        matched_ids, _ = match_predictions_to_gt(pred_results, gt_anns, iou_threshold=0.5)
        matched_gt.update(matched_ids)
        false_positives += len(pred_results) - len(matched_ids)

    false_negatives = total_gt - len(matched_gt)
    mota = 1.0 - (false_negatives + false_positives) / total_gt if total_gt > 0 else 0.0

    return {
        "MOTA": mota,
        "false_negatives": false_negatives,
        "false_positives": false_positives,
        "matched": len(matched_gt),
        "total_gt": total_gt,
    }


def get_video_image_mapping(
    test_dataset, frame_idx_to_unique_image_id: Dict[int, int]
) -> Dict[str, List[int]]:
    """
    Create a mapping from video names to their associated image IDs.

    This is useful for per-video analysis, where you need to know which
    frames (image IDs) belong to which video.

    Args:
        test_dataset: GMIND dataset object with frame_index and video_items attributes
        frame_idx_to_unique_image_id: Dictionary mapping dataset frame index to
                                      unique COCO image ID

    Returns:
        Dictionary mapping video name (string) to list of image IDs (integers).
        Video names are formatted as: "set_name/subdir/video_filename.mp4"

    Example:
        >>> mapping = get_video_image_mapping(test_dataset, frame_idx_to_id)
        >>> video_1_images = mapping["UrbanJunctionSet/1/FLIR8.9-Urban1.mp4"]
        >>> print(f"Video 1 has {len(video_1_images)} frames")
    """
    from collections import defaultdict

    video_image_ids = defaultdict(list)

    for frame_idx, frame_info in enumerate(test_dataset.frame_index):
        video_item = test_dataset.video_items[frame_info["video_idx"]]
        video_path = Path(video_item["video_path"])
        video_name = f"{video_path.parent.parent.name}/{video_path.parent.name}/{video_path.name}"

        image_id = frame_idx_to_unique_image_id.get(frame_idx, frame_idx + 1)
        video_image_ids[video_name].append(image_id)

    return video_image_ids


logger = logging.getLogger(__name__)


def _compute_bbox_distances(
    bboxes: List[Dict],
    id_key: str,
    camera_matrix: np.ndarray,
    camera_height: float,
    camera_pitch_deg: float,
) -> Dict[int, float]:
    """Project each bbox onto the ground plane and return its forward distance.

    Works for both GT annotations and prediction dicts — the only difference
    is which key holds the unique identifier (``"id"`` for GT, we'll use the
    list index for predictions since they don't have unique IDs).

    Args:
        bboxes: List of dicts, each with a ``"bbox"`` key in COCO format
                ``[x, y, w, h]``.
        id_key: Key to use as the unique identifier in the returned dict.
                For GT annotations this is ``"id"``; for predictions we pass
                a synthetic ``"_idx"`` key added by the caller.
        camera_matrix: 3×3 camera intrinsics.
        camera_height: Camera height above ground (metres).
        camera_pitch_deg: Camera pitch (degrees, positive = downward).

    Returns:
        Dict mapping identifier → forward distance in metres.
        Entries where the ray didn't hit the ground plane are omitted.
    """
    from Annotation.footpoint_to_ground import bbox_to_3d_geometric_robust

    distances: Dict[int, float] = {}
    for item in bboxes:
        x, y, w, h = item["bbox"]
        bbox_xyxy = np.array([x, y, x + w, y + h], dtype=np.float32)
        point_3d = bbox_to_3d_geometric_robust(
            bbox_xyxy, camera_matrix, camera_height, camera_pitch_deg
        )
        if point_3d is not None:
            # Y component is forward distance from camera
            distances[item[id_key]] = float(point_3d[1])
    return distances


def _bin_label(bin_idx: int, bin_edges: List[float]) -> str:
    """Return a human-readable label for a ``np.digitize`` bin index."""
    if bin_idx <= 0:
        return f"<{bin_edges[0]}m"
    elif bin_idx >= len(bin_edges):
        return f"{bin_edges[-1]}m+"
    else:
        return f"{bin_edges[bin_idx - 1]}-{bin_edges[bin_idx]}m"


def _assign_bins(
    ids: List[int],
    distances: Dict[int, float],
    bin_edges: List[float],
) -> Dict[str, List[int]]:
    """Group IDs into distance bins.

    Args:
        ids: All candidate IDs (GT annotation IDs or prediction indices).
        distances: Mapping of id → distance.  IDs not present in this dict
                   (ray missed ground) are skipped.
        bin_edges: Sorted bin edge values in metres.

    Returns:
        Dict mapping bin label (e.g. ``"0-20m"``) to list of IDs in that bin.
    """
    bin_edges = sorted(bin_edges)
    bins: Dict[str, List[int]] = defaultdict(list)
    for item_id in ids:
        if item_id not in distances:
            continue
        idx = int(np.digitize(distances[item_id], bin_edges))
        bins[_bin_label(idx, bin_edges)].append(item_id)
    return bins


def compute_distance_binned_metrics(
    coco_gt: COCO,
    coco_results: List[Dict],
    camera_matrix: np.ndarray,
    camera_height: float,
    camera_pitch_deg: float,
    bin_edges: List[float],
    evaluated_img_ids: Optional[List[int]] = None,
    per_class: bool = False,
) -> Dict[str, Dict]:
    """Compute COCO metrics grouped by distance bins.

    Both GT annotations and predictions are projected onto the ground plane
    so that each bin is evaluated with only the GT *and* predictions that
    fall in that distance range.  This avoids false-positive inflation when
    an image contains objects at multiple distances.

    Args:
        coco_gt: COCO ground truth object.
        coco_results: Prediction results in COCO format.
        camera_matrix: 3×3 camera intrinsics.
        camera_height: Camera height above ground (metres).
        camera_pitch_deg: Camera pitch (degrees, positive = downward).
        bin_edges: Sorted distance bin edges in metres,
                   e.g. ``[0, 20, 40, 60, 80, 100]``.
        evaluated_img_ids: List of image IDs that were actually evaluated.
            The COCO GT object may contain annotations for many more images
            than were evaluated (e.g. the full video vs. the validation
            split).  When provided, GT annotations are scoped to only these
            images before computing distances and metrics.

    Returns:
        Dict mapping bin label to a dict with ``"num_gt"`` and the standard
        COCO metrics returned by :func:`compute_coco_metrics`.
    """
    # ---- 0. Scope GT to evaluated images only ----
    # val_dataset.coco may contain annotations for ALL frames in the JSON,
    # not just the validation split.  Restrict to evaluated images so GT
    # counts match what the model actually saw.
    if evaluated_img_ids is not None:
        eval_img_set = set(evaluated_img_ids)
        scoped_anns = {
            aid: ann for aid, ann in coco_gt.anns.items()
            if ann["image_id"] in eval_img_set
        }
        logger.info(
            f"Scoped GT from {len(coco_gt.anns)} to {len(scoped_anns)} "
            f"annotations ({len(eval_img_set)} evaluated images)"
        )
    else:
        scoped_anns = dict(coco_gt.anns)

    # ---- 1. Compute distance for every GT annotation ----
    gt_distances = _compute_bbox_distances(
        list(scoped_anns.values()), "id",
        camera_matrix, camera_height, camera_pitch_deg,
    )
    logger.info(
        f"Distance computed for {len(gt_distances)}/{len(scoped_anns)} "
        f"GT annotations"
    )

    # ---- 2. Compute distance for every prediction ----
    # Predictions don't have a unique "id" key, so we tag each with its
    # list index under a temporary "_idx" key.
    for i, pred in enumerate(coco_results):
        pred["_idx"] = i
    pred_distances = _compute_bbox_distances(
        coco_results, "_idx",
        camera_matrix, camera_height, camera_pitch_deg,
    )
    logger.info(
        f"Distance computed for {len(pred_distances)}/{len(coco_results)} "
        f"predictions"
    )

    # ---- 3. Assign GT and predictions to bins ----
    gt_bins = _assign_bins(
        list(scoped_anns.keys()), gt_distances, bin_edges,
    )
    pred_bins = _assign_bins(
        [p["_idx"] for p in coco_results], pred_distances, bin_edges,
    )

    # ---- 4. Per-bin COCO evaluation ----
    gt_categories = coco_gt.dataset.get("categories", [])
    gt_images = {img_id: info for img_id, info in coco_gt.imgs.items()}

    binned_metrics: Dict[str, Dict] = {}
    # Iterate over all bin labels that appear in *either* GT or predictions
    all_labels = sorted(set(gt_bins.keys()) | set(pred_bins.keys()))
    for label in all_labels:
        ann_ids_in_bin = gt_bins.get(label, [])

        # Collect GT annotations for this bin
        filtered_anns = [scoped_anns[aid] for aid in ann_ids_in_bin]
        gt_image_ids = {ann["image_id"] for ann in filtered_anns}

        # Collect predictions for this bin (strip temporary _idx key)
        pred_indices = set(pred_bins.get(label, []))
        filtered_preds = [
            {k: v for k, v in coco_results[i].items() if k != "_idx"}
            for i in pred_indices
        ]
        pred_image_ids = {p["image_id"] for p in filtered_preds}

        if not filtered_anns:
            logger.warning(f"Bin {label}: no GT annotations")
            binned_metrics[label] = {"num_gt": 0}
            continue

        if not filtered_preds:
            logger.warning(
                f"Bin {label}: {len(ann_ids_in_bin)} GT but no predictions"
            )
            binned_metrics[label] = {"num_gt": len(ann_ids_in_bin)}
            continue

        # Union: loadRes requires every prediction image_id to exist
        # in the GT image list, even if that image has no GT in this bin.
        all_image_ids = gt_image_ids | pred_image_ids
        filtered_gt_dict = {
            "images": [gt_images[iid] for iid in all_image_ids],
            "annotations": filtered_anns,
            "categories": gt_categories,
        }
        filtered_coco_gt = COCO()
        filtered_coco_gt.dataset = filtered_gt_dict
        filtered_coco_gt.createIndex()

        metrics = compute_coco_metrics(filtered_coco_gt, filtered_preds)
        metrics["num_gt"] = len(ann_ids_in_bin)

        if per_class:
            coco_dt = filtered_coco_gt.loadRes(filtered_preds)
            pc = {}
            for m in ("AP50-95", "AP50", "AP75"):
                class_ap = compute_per_class_ap(filtered_coco_gt, coco_dt, metric=m)
                for cls_name, val in class_ap.items():
                    pc.setdefault(cls_name, {})[m] = float(val)
            metrics["per_class"] = pc

        binned_metrics[label] = metrics

        logger.info(
            f"Bin {label}: {len(ann_ids_in_bin)} GT, "
            f"{len(filtered_preds)} preds, "
            f"AP50={metrics.get('AP50', -1):.4f}"
        )

    # Clean up temporary _idx keys from predictions
    for pred in coco_results:
        pred.pop("_idx", None)

    return binned_metrics
