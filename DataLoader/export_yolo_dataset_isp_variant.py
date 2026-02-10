"""Standalone script to export a GMIND dataset to YOLO format.

Usage examples::

    # Basic export
    python -m DataLoader.export_yolo_dataset \
        --gmind-config DeepLearning/gmind_config.yaml \
        --out-dir /path/to/yolo_dataset

    # With ISP variant — writes to /path/to/yolo_datasets/Default_ISP/
    python -m DataLoader.export_yolo_dataset \
        --gmind-config SensitivityAnalysis/sensitivity_config.yaml \
        --isp-variant Default_ISP \
        --out-dir /path/to/yolo_datasets
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

from DataLoader import get_gmind_dataloader
from DataLoader.export_to_yolo import export_gmind_to_yolo

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a GMIND (or ISP-variant) dataset to YOLO format."
    )
    parser.add_argument(
        "--gmind-config",
        type=Path,
        required=True,
        help="Path to the YAML config file (e.g. gmind_config.yaml)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for the YOLO dataset",
    )
    parser.add_argument(
        "--isp-variant",
        type=str,
        default=None,
        help="ISP variant name (uses ISPVariantDataset instead of GMINDDataset)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # ------------------------------------------------------------------
    # Load config
    # ------------------------------------------------------------------
    config_path: Path = args.gmind_config
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data_root = Path(config["data"]["root"])
    sensor = config["data"].get("sensor", "FLIR8.9")
    frame_stride = config["data"].get("frame_stride", 1)
    max_frames_per_video = config["data"].get("max_frames_per_video")

    # ------------------------------------------------------------------
    # Choose dataloader factory
    # ------------------------------------------------------------------
    if args.isp_variant:
        from SensitivityAnalysis.isp_dataset import get_isp_dataloader as _get_loader

        logger.info("ISP variant mode: '%s'", args.isp_variant)
    else:
        _get_loader = get_gmind_dataloader

    isp_kwargs = {"isp_variant": args.isp_variant} if args.isp_variant else {}

    # ------------------------------------------------------------------
    # Build train dataset
    # ------------------------------------------------------------------
    train_config = config.get("train", {})
    train_sets = train_config.get("sets", ["UrbanJunctionSet"])
    train_max_frames = train_config.get("max_frames_per_video", max_frames_per_video)
    train_set_subdirs = train_config.get("set_subdirs", {})
    train_percentage_split = train_config.get("percentage_split", {})
    train_percentage_split_start = train_config.get("percentage_split_start", {})

    logger.info("Loading train dataset: sets=%s, sensor=%s", train_sets, sensor)
    train_dataset = _get_loader(
        data_root=data_root,
        sets=train_sets,
        sensor=sensor,
        transforms=None,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        frame_stride=frame_stride,
        max_frames=train_max_frames,
        set_subdirs=train_set_subdirs,
        percentage_split=train_percentage_split,
        percentage_split_start=train_percentage_split_start,
        **isp_kwargs,
    ).dataset

    # ------------------------------------------------------------------
    # Build validation dataset
    # ------------------------------------------------------------------
    val_config = config.get("validation", {})
    val_sets = val_config.get("sets", train_sets)
    val_max_frames = val_config.get("max_frames_per_video", max_frames_per_video)
    val_set_subdirs = val_config.get("set_subdirs", train_set_subdirs)
    val_percentage_split = val_config.get("percentage_split", train_percentage_split)
    val_percentage_split_start = val_config.get(
        "percentage_split_start", train_percentage_split_start
    )

    logger.info("Loading validation dataset: sets=%s, sensor=%s", val_sets, sensor)
    val_dataset = _get_loader(
        data_root=data_root,
        sets=val_sets,
        sensor=sensor,
        transforms=None,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        frame_stride=frame_stride,
        max_frames=val_max_frames,
        set_subdirs=val_set_subdirs,
        percentage_split=val_percentage_split,
        percentage_split_start=val_percentage_split_start,
        **isp_kwargs,
    ).dataset

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    out_dir: Path = args.out_dir
    if args.isp_variant:
        out_dir = out_dir / args.isp_variant
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Exporting to YOLO format: %s  (train=%d, val=%d)",
        out_dir,
        len(train_dataset),
        len(val_dataset),
    )
    data_yaml = export_gmind_to_yolo(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        out_dir=str(out_dir),
    )
    logger.info("Export complete. data.yaml: %s", data_yaml)


if __name__ == "__main__":
    main()
