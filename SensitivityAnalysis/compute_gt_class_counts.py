"""Compute per-class GT instance counts per distance bin for the test split.

Reads the COCO ground-truth JSON (e.g. ``sensitivity_results/coco_gt.json``),
filters to test-split image IDs using the same dataset config as the eval
pipeline, projects each GT bbox onto the ground plane, and counts instances
per (class, distance-bin) pair.

Example
-------
python -m SensitivityAnalysis.compute_gt_class_counts \
    --gt-file sensitivity_results/coco_gt.json \
    --config SensitivityAnalysis/sensitivity_config.yaml \
    --isp-variant Default_ISP \
    --frame-stride 5 \
    --bin-edges 0 15 30 45 60 75 \
    --output sensitivity_results/gt_class_bin_counts.json
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from pycocotools.coco import COCO

from Annotation.annotation_generation import parse_camera_intrinsics_from_calibration
from Evaluation.analysis.analysis_utils import (
    _assign_bins,
    _bin_label,
    _compute_bbox_distances,
)
from SensitivityAnalysis.isp_dataset import ISPVariantDataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def _build_test_dataset(config: dict, isp_variant: str, frame_stride: int) -> ISPVariantDataset:
    """Instantiate an ISPVariantDataset for the test split."""
    data_cfg = config["data"]
    test_cfg = config["test"]

    return ISPVariantDataset(
        data_root=data_cfg["root"],
        isp_variant=isp_variant,
        sets=test_cfg["sets"],
        sensor=data_cfg["sensor"],
        annotation_format="coco",
        transforms=None,
        frame_stride=frame_stride,
        max_frames=data_cfg.get("max_frames_per_video"),
        set_subdirs=test_cfg.get("set_subdirs"),
        percentage_split=test_cfg.get("percentage_split"),
        percentage_split_start=test_cfg.get("percentage_split_start"),
    )


def _extract_test_image_ids(dataset: ISPVariantDataset) -> list[int]:
    """Collect COCO image IDs for the test split frames."""
    image_ids = []
    for entry in dataset.frame_index:
        image_info = entry.get("image_info")
        if image_info is not None:
            image_ids.append(image_info["id"])
    return image_ids


def compute_gt_class_bin_counts(
    gt_file: Path,
    config: dict,
    isp_variant: str,
    frame_stride: int,
    bin_edges: list[float],
) -> dict:
    """Compute per-class GT counts per distance bin for the test split.

    Returns a dict ready for JSON serialisation.
    """
    # Normalise bin edges: 15.0 -> 15 so labels match eval results ("0-15m" not "0.0-15.0m")
    bin_edges = [int(e) if float(e).is_integer() else e for e in bin_edges]

    # 1. Build test dataset to get image IDs
    dataset = _build_test_dataset(config, isp_variant, frame_stride)
    test_image_ids = _extract_test_image_ids(dataset)
    log.info("Test split: %d images", len(test_image_ids))

    # 2. Load COCO GT and filter to test images
    coco_gt = COCO(str(gt_file))
    test_id_set = set(test_image_ids)
    ann_ids = coco_gt.getAnnIds(imgIds=list(test_id_set))
    test_anns = coco_gt.loadAnns(ann_ids)
    log.info("GT annotations in test split: %d", len(test_anns))

    # 3. Parse camera intrinsics
    project_root = Path(__file__).resolve().parent.parent
    calib_path = project_root / "sensor_calibration.txt"
    sensor = config["data"]["sensor"]
    camera_matrix, _ = parse_camera_intrinsics_from_calibration(
        str(calib_path), camera_name=sensor,
    )
    if camera_matrix is None:
        raise RuntimeError(f"Failed to parse camera intrinsics from {calib_path}")

    # 4. Compute distances for all test GT annotations
    dist_cfg = config["distance_eval"]
    gt_distances = _compute_bbox_distances(
        test_anns,
        "id",
        camera_matrix,
        dist_cfg["camera_height"],
        dist_cfg["camera_pitch_deg"],
    )
    log.info(
        "Distance computed for %d / %d GT annotations",
        len(gt_distances), len(test_anns),
    )

    # 5. Assign to bins
    gt_bins = _assign_bins(
        [ann["id"] for ann in test_anns],
        gt_distances,
        bin_edges,
    )

    # 6. Build category_id -> name mapping
    cats = {c["id"]: c["name"] for c in coco_gt.dataset["categories"]}
    ann_by_id = {ann["id"]: ann for ann in test_anns}

    # 7. Count per (class, bin)
    sorted_edges = sorted(bin_edges)
    all_bin_labels = [_bin_label(i, sorted_edges) for i in range(1, len(sorted_edges))]

    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    total_per_bin: dict[str, int] = defaultdict(int)

    for bin_label, ann_ids_in_bin in gt_bins.items():
        for aid in ann_ids_in_bin:
            ann = ann_by_id[aid]
            cls_name = cats[ann["category_id"]]
            counts[cls_name][bin_label] += 1
            total_per_bin[bin_label] += 1

    # Ensure all bins appear for all classes (fill zeros)
    all_classes = sorted(counts.keys())
    for cls in all_classes:
        for bl in all_bin_labels:
            counts[cls].setdefault(bl, 0)
    for bl in all_bin_labels:
        total_per_bin.setdefault(bl, 0)

    return {
        "bin_edges": sorted_edges,
        "bins": all_bin_labels,
        "counts": {cls: dict(counts[cls]) for cls in all_classes},
        "total_per_bin": dict(total_per_bin),
        "num_test_images": len(test_image_ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute per-class GT instance counts per distance bin.",
    )
    parser.add_argument(
        "--gt-file", type=Path, required=True,
        help="Path to coco_gt.json",
    )
    parser.add_argument(
        "--config", type=Path, required=True,
        help="Path to sensitivity_config.yaml",
    )
    parser.add_argument(
        "--isp-variant", type=str, default="Default_ISP",
        help="ISP variant name (default: Default_ISP)",
    )
    parser.add_argument(
        "--frame-stride", type=int, default=5,
        help="Frame stride for test split (default: 5)",
    )
    parser.add_argument(
        "--bin-edges", type=float, nargs="+", required=True,
        help="Distance bin edges in metres (e.g. 0 15 30 45 60 75)",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output JSON file path",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    result = compute_gt_class_bin_counts(
        gt_file=args.gt_file,
        config=config,
        isp_variant=args.isp_variant,
        frame_stride=args.frame_stride,
        bin_edges=args.bin_edges,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    log.info("Saved: %s", args.output)

    # Print summary
    log.info("Test images: %d", result["num_test_images"])
    log.info("Bins: %s", result["bins"])
    for cls, bin_counts in result["counts"].items():
        total = sum(bin_counts.values())
        log.info("  %s: %d total  %s", cls, total, bin_counts)


if __name__ == "__main__":
    main()
