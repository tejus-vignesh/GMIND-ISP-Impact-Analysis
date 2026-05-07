#!/usr/bin/env python3
"""Generate a COCO ground-truth JSON file from the GMIND dataset.

The resulting ``coco_gt.json`` can be used together with an
``eval_results_detections.json`` (produced by ``train_models.py --eval-only``)
to visualise predictions alongside ground truth using
``visualise_gt_and_pred.py``.

Usage:
    python -m Evaluation.visualisation.generate_coco_gt \
        --gmind-config SensitivityAnalysis/sensitivity_config.yaml \
        --isp-variant Default_ISP \
        --split test \
        --output ./results/fasterrcnn_resnet50_fpn/Default_ISP/coco_gt.json
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def main():
    parser = argparse.ArgumentParser(
        description="Generate COCO ground-truth JSON from the GMIND dataset"
    )
    parser.add_argument(
        "--gmind-config",
        type=str,
        required=True,
        help="Path to GMIND config YAML (e.g. SensitivityAnalysis/sensitivity_config.yaml)",
    )
    parser.add_argument(
        "--isp-variant",
        type=str,
        default=None,
        help="ISP variant name (e.g. Default_ISP). Omit for standard GMIND dataset.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "validation", "test"],
        help="Dataset split to generate GT for (default: test)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="coco_gt.json",
        help="Output path for the COCO GT JSON file (default: coco_gt.json)",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=None,
        help="Override frame_stride from config",
    )
    args = parser.parse_args()

    # Load config
    config_path = Path(args.gmind_config)
    if not config_path.exists():
        print(f"Error: config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data_root = Path(config["data"]["root"])
    sensor = config["data"].get("sensor", "FLIR8.9")
    frame_stride = args.frame_stride if args.frame_stride is not None else config["data"].get("frame_stride", 1)
    max_frames_per_video = config["data"].get("max_frames_per_video")

    # Read split-specific config
    split_config = config.get(args.split, {})
    sets = split_config.get("sets", [])
    set_subdirs = split_config.get("set_subdirs", {})
    percentage_split = split_config.get("percentage_split", {})
    percentage_split_start = split_config.get("percentage_split_start", {})
    max_frames = split_config.get("max_frames_per_video", max_frames_per_video)

    if not sets:
        print(f"Error: no sets defined for split '{args.split}' in config")
        sys.exit(1)

    # Choose dataset class based on --isp-variant
    if args.isp_variant:
        from SensitivityAnalysis.isp_dataset import ISPVariantDataset as DatasetClass

        extra_kwargs = {"isp_variant": args.isp_variant}
        print(f"Using ISPVariantDataset with variant '{args.isp_variant}'")
    else:
        from DataLoader.gmind_dataset import GMINDDataset as DatasetClass

        extra_kwargs = {}
        print("Using GMINDDataset")

    print(f"Split: {args.split}")
    print(f"Sets: {sets}")
    print(f"Sensor: {sensor}")

    # Build dataset (this triggers _build_frame_index + _build_coco_ground_truth)
    dataset = DatasetClass(
        data_root=data_root,
        sets=sets,
        sensor=sensor,
        transforms=None,
        frame_stride=frame_stride,
        max_frames=max_frames,
        set_subdirs=set_subdirs,
        percentage_split=percentage_split,
        percentage_split_start=percentage_split_start,
        **extra_kwargs,
    )

    if dataset.coco is None:
        print("Error: COCO ground truth was not built (pycocotools not installed?)")
        sys.exit(1)

    coco_dict = dataset.coco.dataset
    print(
        f"COCO GT: {len(coco_dict['images'])} images, "
        f"{len(coco_dict['annotations'])} annotations, "
        f"{len(coco_dict['categories'])} categories"
    )

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(coco_dict, f)

    print(f"Saved COCO GT to: {output_path}")


if __name__ == "__main__":
    main()
