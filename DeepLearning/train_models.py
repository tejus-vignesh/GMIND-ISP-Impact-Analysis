"""Production-ready PyTorch training script for object detection models.

This module provides a comprehensive training framework supporting multiple backends:
- TorchVision models (Faster R-CNN, Mask R-CNN, RetinaNet, SSD, etc.)
- Ultralytics YOLO models (YOLOv5, YOLOv8, YOLOv9, YOLOv10, YOLOv11, YOLOv12, RT-DETR)
- MMDetection models (Faster R-CNN, DETR, YOLOX, and 50+ architectures)

Features:
- Multi-backend support with automatic backend detection
- COCO dataset support (standard format)
- GMIND dataset support (via DataLoader module)
- Mixed precision training (AMP)
- Checkpoint management (save/resume)
- COCO evaluation metrics
- TensorBoard logging
- Comprehensive CLI interface

Example usage:
    # Train a TorchVision model on COCO
    python -m DeepLearning.train_models --data /path/to/coco --model fasterrcnn_resnet50_fpn --epochs 12

    # Train a YOLO model on GMIND dataset
    python -m DeepLearning.train_models --use-gmind --gmind-config config.yaml --model yolov8m --epochs 50

    # Evaluate only
    python -m DeepLearning.train_models --use-gmind --eval-only --eval-checkpoint checkpoint.pth
"""

import argparse
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CocoDetection
from tqdm import tqdm

# Configure logging first
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# TensorBoard support
try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    logger.warning("TensorBoard not available. Install with: pip install tensorboard")

try:
    from pycocotools import mask as coco_mask
except ImportError:  # pragma: no cover - optional dependency
    coco_mask = None


def collate_fn(batch):
    """Collate function for DataLoader batches.

    Args:
        batch: List of (image, target) tuples

    Returns:
        Tuple of (images_list, targets_list)
    """
    images, targets = zip(*batch)
    return list(images), list(targets)


def get_transform(train: bool):
    """Get detection-aware transforms for training or validation.

    Returns transforms that operate on (image, target) pairs, ensuring
    augmentations like horizontal flip are applied to both image and
    bounding box coordinates.

    Uses detection transforms from DeepLearning.augmentations module.
    """
    from DeepLearning.augmentations import get_detection_transforms
    return get_detection_transforms(train=train, augment_level="light")


DEFAULT_COCO_KEYPOINTS = 17


def _convert_segmentation_to_mask(
    segmentation: Any, height: int, width: int
) -> Optional[np.ndarray]:
    if coco_mask is None or segmentation is None:
        return None

    try:
        if isinstance(segmentation, list) and segmentation:
            rles = coco_mask.frPyObjects(segmentation, height, width)
            mask = coco_mask.decode(rles)
            if isinstance(mask, list):
                mask = np.stack(mask, axis=2)
            mask = mask.any(axis=2)
        elif isinstance(segmentation, dict) and segmentation:
            mask = coco_mask.decode(segmentation)
            if mask.ndim == 3:
                mask = mask[..., 0]
        else:
            return None
    except Exception:
        return None

    return mask.astype(np.uint8)


def coco_to_target(coco_ann: List[Dict[str, Any]], image_id: int, image_size: Tuple[int, int]):
    width, height = image_size
    boxes, labels, areas, iscrowd = [], [], [], []
    masks: List[torch.Tensor] = []
    keypoints: List[torch.Tensor] = []

    for ann in coco_ann:
        x, y, w, h = ann["bbox"]
        boxes.append([x, y, x + w, y + h])
        labels.append(ann.get("category_id", 0))
        areas.append(ann.get("area", w * h))
        iscrowd.append(ann.get("iscrowd", 0))
        mask_array = _convert_segmentation_to_mask(ann.get("segmentation"), height, width)
        if mask_array is None:
            masks.append(torch.zeros((height, width), dtype=torch.uint8))
        else:
            masks.append(torch.as_tensor(mask_array, dtype=torch.uint8))

        kp = ann.get("keypoints")
        if kp:
            kp_tensor = torch.as_tensor(kp, dtype=torch.float32).reshape(-1, 3)
        else:
            kp_tensor = torch.zeros((DEFAULT_COCO_KEYPOINTS, 3), dtype=torch.float32)
        keypoints.append(kp_tensor)

    if boxes:
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        masks = torch.stack(masks) if masks else torch.zeros((0, height, width), dtype=torch.uint8)
        keypoints = (
            torch.stack(keypoints)
            if keypoints
            else torch.zeros((0, DEFAULT_COCO_KEYPOINTS, 3), dtype=torch.float32)
        )
    else:
        boxes = torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.zeros((0,), dtype=torch.int64)
        areas = torch.zeros((0,), dtype=torch.float32)
        iscrowd = torch.zeros((0,), dtype=torch.int64)
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
        keypoints = torch.zeros((0, DEFAULT_COCO_KEYPOINTS, 3), dtype=torch.float32)

    return {
        "boxes": boxes,
        "labels": labels,
        "image_id": torch.tensor([image_id]),
        "area": areas,
        "iscrowd": iscrowd,
        "masks": masks,
        "keypoints": keypoints,
    }


class CocoDetectionWrapper(CocoDetection):
    def __init__(self, img_folder, ann_file, transforms=None):
        super().__init__(img_folder, ann_file)
        self._transforms = transforms

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        image_id = self.ids[idx]
        w, h = img.size
        target = coco_to_target(target, image_id, (w, h))
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


def get_model(model_name: str, num_classes: int, pretrained: bool = True, backend: str = "auto"):
    """
    Load a detection model from any supported backend (TorchVision, Ultralytics, MMDetection).

    Automatically detects the backend from the model name if not specified.

    Args:
        model_name: Model identifier (e.g., 'fasterrcnn_resnet50_fpn', 'yolov8m', 'faster_rcnn')
        num_classes: Number of output classes (including background)
        pretrained: Whether to use pretrained weights
    backend: Backend to use ('auto' for auto-detection, or explicit:
        'torchvision', 'ultralytics', 'mmdetection')

    Returns:
        Configured model object from the appropriate backend

    Raises:
        ValueError: If model is not available or backend is not installed

    Examples:
        >>> # Auto-detect backend (recommended)
        >>> model = get_model('fasterrcnn_resnet50_fpn', num_classes=80)
        >>>
        >>> # Explicit backend
        >>> model = get_model('yolov8m', num_classes=80, backend='ultralytics')
    """
    from DeepLearning.adapters import get_model as adapter_get_model

    try:
        model = adapter_get_model(
            model_name=model_name, num_classes=num_classes, pretrained=pretrained, backend=backend
        )
        return model
    except Exception as e:
        logger.error(f"Failed to load model '{model_name}': {e}")
        raise


def get_supported_models() -> List[str]:
    return [
        "fasterrcnn_resnet50_fpn",
        "fasterrcnn_mobilenet_v3_large_fpn",
        "maskrcnn_resnet50_fpn",
        "ssd300_vgg16",
        "ssdlite320_mobilenet_v3_large",
    ]


def save_checkpoint(state: dict, checkpoint_dir: str, filename: str):
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    path = Path(checkpoint_dir) / filename
    torch.save(state, str(path))
    logger.info(f"Saved checkpoint: {path}")


class _YOLOTorchVisionAdapter:
    """Wrap an Ultralytics YOLO model so it can be used with evaluate_coco().

    evaluate_coco() expects a model that:
      - Has an `eval()` method
      - Is callable with a list of image tensors
      - Returns a list of dicts with "boxes", "scores", "labels" keys

    This adapter converts Ultralytics Result objects to that format.
    """

    def __init__(self, yolo_model):
        self.model = yolo_model

    def eval(self):
        self.model.eval()

    def __call__(self, images):
        np_images = [
            img.mul(255).byte().permute(1, 2, 0).cpu().numpy()
            for img in images
        ]
        results = self.model(np_images, verbose=False)
        outputs = []
        for r in results:
            boxes = r.boxes
            outputs.append({
                "boxes": boxes.xyxy,       # already [x1,y1,x2,y2] tensor
                "scores": boxes.conf,
                "labels": boxes.cls.int() + 1,  # YOLO 0-indexed → COCO 1-indexed
            })
        return outputs


def evaluate_coco(
    model,
    data_loader,
    device,
    val_dataset=None,
    save_results: Optional[str] = None,
    subset: Optional[int] = None,
) -> Tuple[Optional[list], List[Dict], Optional[Any], List[int]]:
    """Run COCO evaluation (bbox) using pycocotools if available.

    Args:
        model: Detection model to evaluate
        data_loader: DataLoader for validation data
        device: Device to run evaluation on
        val_dataset: Validation dataset (optional, will try to extract from data_loader)
        save_results: Path to save detection results JSON (COCO format)
        subset: Only evaluate on subset of images (for quick tests)

    Returns:
        Tuple of (stats, results, coco_gt, img_ids):
        - stats: COCOeval.stats array if successful, otherwise None
        - results: List of prediction dicts in COCO results format
        - coco_gt: COCO ground truth object, or None if unavailable
        - img_ids: List of all image IDs that were evaluated (includes
          frames with zero detections)
    """
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        logger.warning("pycocotools not installed; skipping COCO evaluation.")
        return None, [], None, []

    model.eval()
    results = []
    img_ids = []
    processed = 0

    for images, targets in tqdm(data_loader, desc="Eval", unit="img"):
        images = [img.to(device) for img in images]
        with torch.no_grad():
            outputs = model(images)

        for out, tgt in zip(outputs, targets):
            # Extract image ID from target
            if isinstance(tgt["image_id"], torch.Tensor):
                image_id = int(tgt["image_id"].item())
            else:
                image_id = int(tgt["image_id"])

            # Extract detections
            boxes = out.get("boxes", torch.empty((0, 4))).cpu().numpy()
            scores = out.get("scores", torch.empty((0,))).cpu().numpy()
            labels = (
                out.get("labels", torch.empty((0,), dtype=torch.int64)).cpu().numpy()
                if "labels" in out
                else None
            )

            if labels is None:
                continue

            # Convert to COCO format
            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box.tolist()
                w = x2 - x1
                h = y2 - y1
                results.append(
                    {
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": [float(x1), float(y1), float(w), float(h)],
                        "score": float(score),
                    }
                )
            img_ids.append(image_id)

        processed += len(images)
        if subset is not None and processed >= subset:
            break

    # Get ground truth COCO object (also guard against .coco being None)
    cocoGt = None
    if val_dataset is not None and getattr(val_dataset, "coco", None) is not None:
        cocoGt = val_dataset.coco
    elif getattr(data_loader.dataset, "coco", None) is not None:
        cocoGt = data_loader.dataset.coco

    if cocoGt is None:
        logger.warning("COCO ground truth not available in dataset; cannot run full COCO eval.")
        # Still save results if requested
        if save_results and results:
            with open(save_results, "w") as f:
                json.dump(results, f)
        return None, results, None, list(sorted(set(img_ids)))

    if len(results) == 0:
        logger.warning("No detections produced; skipping COCO eval.")
        # Still save empty results if requested
        if save_results:
            with open(save_results, "w") as f:
                json.dump(results, f)
        return None, [], cocoGt, list(sorted(set(img_ids)))

    # Save raw results if requested
    if save_results:
        with open(save_results, "w") as f:
            json.dump(results, f)
        logger.info(f"Saved detection results to {save_results}")

    # Run COCO evaluation
    cocoDt = cocoGt.loadRes(results)
    cocoEval = COCOeval(cocoGt, cocoDt, iouType="bbox")
    cocoEval.params.imgIds = list(sorted(set(img_ids)))
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    return cocoEval.stats, results, cocoGt, list(sorted(set(img_ids)))


def load_checkpoint(
    model: nn.Module, optimizer: torch.optim.Optimizer, path: str, device: torch.device
):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_epoch = checkpoint.get("epoch", 0) + 1
    scaler_state = checkpoint.get("scaler_state", None)
    logger.info(f"Loaded checkpoint '{path}' (epoch {start_epoch-1})")
    return start_epoch, scaler_state


def load_model_weights(model: nn.Module, path: str, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict):
        if "model_state" in checkpoint:
            state = checkpoint["model_state"]
        elif "state_dict" in checkpoint:
            state = checkpoint["state_dict"]
        else:
            state = checkpoint
    else:
        state = checkpoint
    model.load_state_dict(state)
    logger.info(f"Loaded model weights from '{path}'")


def train_one_epoch(
    model, optimizer, dataloader, device, epoch, scaler=None, print_freq=50
) -> float:
    model.train()
    metric_logger = tqdm(dataloader, desc=f"Epoch {epoch}", unit="batch")
    running_loss = 0.0
    for batch_idx, (images, targets) in enumerate(metric_logger):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=(scaler is not None)):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        running_loss += loss_value
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        avg_loss = running_loss / (batch_idx + 1)
        metric_logger.set_postfix(loss=f"{avg_loss:.4f}")
    avg_epoch_loss = running_loss / max(1, len(dataloader))
    return avg_epoch_loss


def main():
    parser = argparse.ArgumentParser(description="Train detection models (PyTorch / torchvision)")
    parser.add_argument(
        "--data",
        required=False,
        help="Path to COCO dataset root (required for COCO, not needed for GMIND)",
    )
    parser.add_argument(
        "--model",
        default="fasterrcnn_resnet50_fpn",
        help=f"Model name. Supported: {', '.join(get_supported_models())}",
    )
    parser.add_argument(
        "--backend",
        default="torchvision",
        choices=["torchvision", "ultralytics", "mmdet", "auto"],
        help="Backend for model construction",
    )
    parser.add_argument("--backend-config", default=None, help="Optional backend config file")
    parser.add_argument(
        "--backend-weights", default=None, help="Optional backend weights/checkpoint file"
    )
    parser.add_argument("--epochs", type=int, default=12)
    import torch

    def get_adaptive_batch_size():
        if torch.cuda.is_available():
            try:
                total_mem = torch.cuda.get_device_properties(0).total_memory
                # Use larger batch size for GPUs with >8GB, else small
                if total_mem > 8 * 1024**3:
                    return 8
                elif total_mem > 4 * 1024**3:
                    return 4
                else:
                    return 2
            except Exception:
                return 2
        return 2

    parser.add_argument("--batch-size", type=int, default=get_adaptive_batch_size())
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint-dir", default="./checkpoints")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument(
        "--num-classes", type=int, default=None, help="Number of classes (including background)"
    )
    parser.add_argument(
        "--use-gmind", action="store_true", help="Use GMIND DataLoader instead of COCO"
    )
    parser.add_argument(
        "--gmind-config", default=None, help="Path to GMIND dataset config YAML file"
    )
    parser.add_argument(
        "--isp-variant", default=None,
        help="ISP variant name for sensitivity analysis (e.g., gac_gain-8, Default_ISP, Bayer_GC)",
    )
    parser.add_argument("--use-amp", action="store_true", help="Use mixed precision training")
    parser.add_argument(
        "--do-eval",
        action="store_true",
        help="Run COCO evaluation on validation set after each epoch",
    )
    parser.add_argument(
        "--eval-only", action="store_true", help="Run inference/evaluation on validation set only"
    )
    parser.add_argument(
        "--eval-checkpoint", default=None, help="Checkpoint file for evaluation-only mode"
    )
    parser.add_argument("--eval-output", default=None, help="Path to write detection results JSON")
    parser.add_argument(
        "--eval-subset", type=int, default=None, help="Only run inference on subset of images"
    )
    parser.add_argument(
        "--bin-distance",
        action="store_true",
        help="Enable distance-binned evaluation metrics (requires distance_eval config in sensitivity_config.yaml)",
    )
    parser.add_argument(
        "--augment-level",
        default="light",
        choices=["light", "medium", "heavy"],
        help="Augmentation level for Ultralytics training (light/medium/heavy)",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Auto-structure checkpoint directory: {base}/{model}/{variant}/
    if args.isp_variant:
        checkpoint_dir = Path(args.checkpoint_dir) / args.model / args.isp_variant
    else:
        checkpoint_dir = Path(args.checkpoint_dir)

    # Load datasets - either GMIND or COCO
    if args.use_gmind:
        logger.info("Using GMIND DataLoader")
        try:
            import yaml
        except ImportError:
            logger.error("PyYAML is required for GMIND config. Install with: pip install pyyaml")
            return

        # Load config
        if args.gmind_config:
            config_path = Path(args.gmind_config)
        else:
            # Try default config location
            config_path = Path(__file__).parent / "gmind_config.yaml"

        if not config_path.exists():
            logger.error(f"GMIND config file not found: {config_path}")
            logger.info("Please create a config file or specify with --gmind-config")
            return

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        data_root = Path(config["data"]["root"])
        sensor = config["data"].get("sensor", "FLIR8.9")
        frame_stride = config["data"].get("frame_stride", 1)
        max_frames_per_video = config["data"].get("max_frames_per_video")

        # Import GMIND DataLoader
        if args.isp_variant:
            from SensitivityAnalysis.isp_dataset import get_isp_dataloader as get_gmind_dataloader
            logger.info(f"ISP variant mode: '{args.isp_variant}'")
        else:
            from DataLoader import get_gmind_dataloader

        isp_kwargs = {"isp_variant": args.isp_variant} if args.isp_variant else {}

        # Create train dataset
        train_config = config.get("train", {})
        train_sets = train_config.get("sets", ["UrbanJunctionSet"])
        train_max_frames = train_config.get("max_frames_per_video", max_frames_per_video)
        train_set_subdirs = train_config.get("set_subdirs", {})
        train_percentage_split = train_config.get("percentage_split", {})
        train_percentage_split_start = train_config.get("percentage_split_start", {})

        logger.info(f"Loading GMIND train dataset: sets={train_sets}, sensor={sensor}")
        train_loader = get_gmind_dataloader(
            data_root=data_root,
            sets=train_sets,
            sensor=sensor,
            transforms=get_transform(train=True),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            frame_stride=frame_stride,
            max_frames=train_max_frames,
            set_subdirs=train_set_subdirs,
            percentage_split=train_percentage_split,
            percentage_split_start=train_percentage_split_start,
            **isp_kwargs,
        )
        train_dataset = train_loader.dataset

        # Create validation dataset
        val_config = config.get("validation", {})
        val_sets = val_config.get("sets", train_sets)  # Default to train sets if not specified
        val_max_frames = val_config.get("max_frames_per_video", max_frames_per_video)
        val_set_subdirs = val_config.get("set_subdirs", train_set_subdirs)
        val_percentage_split = val_config.get("percentage_split", train_percentage_split)
        val_percentage_split_start = val_config.get(
            "percentage_split_start", train_percentage_split_start
        )

        logger.info(f"Loading GMIND validation dataset: sets={val_sets}, sensor={sensor}")
        val_loader = get_gmind_dataloader(
            data_root=data_root,
            sets=val_sets,
            sensor=sensor,
            transforms=get_transform(train=False),
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            frame_stride=frame_stride,
            max_frames=val_max_frames,
            set_subdirs=val_set_subdirs,
            percentage_split=val_percentage_split,
            percentage_split_start=val_percentage_split_start,
            **isp_kwargs,
        )
        val_dataset = val_loader.dataset

        # Infer num_classes from dataset
        if args.num_classes is None:
            # Try to get from annotation file
            try:
                import json

                # Find a JSON annotation file
                for set_name in train_sets:
                    set_dir = data_root / set_name
                    if set_dir.exists():
                        for subdir in set_dir.iterdir():
                            if subdir.is_dir() and subdir.name.isdigit():
                                json_files = list(subdir.glob(f"{sensor}-*.json"))
                                if json_files:
                                    with open(json_files[0], "r") as f:
                                        ann_data = json.load(f)
                                    categories = ann_data.get("categories", [])
                                    if categories:
                                        max_cat = max([c["id"] for c in categories])
                                        args.num_classes = max_cat + 1
                                        logger.info(
                                            f"Inferred num_classes={args.num_classes} from GMIND annotations"
                                        )
                                        break
                        if args.num_classes:
                            break
            except Exception as e:
                logger.warning(f"Could not infer num_classes from GMIND annotations: {e}")
                args.num_classes = 4  # Default: person, bicycle, car + background
                logger.info(f"Using default num_classes={args.num_classes}")

    else:
        # Original COCO dataset loading
        if not args.data:
            logger.error("--data argument is required when not using --use-gmind")
            return
        data_root = Path(args.data)
        train_img_folder = data_root / "train2017"
        val_img_folder = data_root / "val2017"
        train_ann = data_root / "annotations" / "instances_train2017.json"
        val_ann = data_root / "annotations" / "instances_val2017.json"

        # If train2017 is missing, use val2017 for both train and val (for testing only)
        if not train_img_folder.exists() or not train_ann.exists():
            logger.warning(
                f"COCO train2017 not found, using val2017 for both training and validation (testing mode)"
            )
            train_img_folder = val_img_folder
            train_ann = val_ann

        logger.info(f"Loading COCO datasets from {data_root}")
        train_dataset = CocoDetectionWrapper(
            str(train_img_folder), str(train_ann), transforms=get_transform(train=True)
        )
        val_dataset = CocoDetectionWrapper(
            str(val_img_folder), str(val_ann), transforms=get_transform(train=False)
        )

    # Limit number of samples for quick tests (only for COCO, GMIND uses config)
    if not args.use_gmind:
        max_train_samples = 40
        max_val_samples = 40
        if len(train_dataset) > max_train_samples:
            import torch.utils.data as data

            train_dataset = data.Subset(train_dataset, range(max_train_samples))
        if len(val_dataset) > max_val_samples:
            import torch.utils.data as data

            val_dataset = data.Subset(val_dataset, range(max_val_samples))

    if args.num_classes is None and not args.use_gmind:
        try:
            with open(train_ann, "r") as f:
                ann = json.load(f)
            categories = ann.get("categories", [])
            max_cat = max([c["id"] for c in categories]) if categories else 0
            args.num_classes = max_cat + 1
            logger.info(f"Inferred num_classes={args.num_classes} from annotations")
        except Exception as e:
            args.num_classes = 91
            logger.warning(f"Could not infer num_classes: {e}. Using default 91.")
    # Detect if this is an Ultralytics model
    # Note: YOLOv12 may not be available in all Ultralytics versions
    # If yolov12 fails, try yolov8 or yolov11 as fallback
    is_ultralytics = args.backend in ("ultralytics", "yolo") or (
        args.backend == "auto" and any(x in args.model.lower() for x in ["yolov", "yolo1", "yolo2", "yolo3", "rtdetr"])
    )

    logger.info(f"Building model: {args.model} with {args.num_classes} classes")
    if args.backend == "torchvision":
        model = get_model(args.model, args.num_classes)
    else:
        from DeepLearning.adapters import detect_backend
        from DeepLearning.adapters import get_model as adapter_get_model

        detected_backend = None
        if args.backend == "auto":
            detected_backend = detect_backend(args.model)
            is_ultralytics = detected_backend == "ultralytics"
        # Build kwargs for adapter (filter out None and unsupported args)
        adapter_kwargs = {"pretrained": False}
        if args.backend_weights:
            adapter_kwargs["weights"] = args.backend_weights
        # Note: config_file is not supported by Ultralytics, only MMDetection

        model = adapter_get_model(
            args.model,
            args.num_classes,
            backend=args.backend if args.backend != "auto" else detected_backend,
            **adapter_kwargs,
        )

    if not is_ultralytics:
        model.to(device)
    logger.info(f"Model loaded successfully (Ultralytics: {is_ultralytics})")

    if args.eval_only and is_ultralytics:
        logger.info("Running in YOLO evaluation-only mode")
        from ultralytics import YOLO

        # Load model from checkpoint
        if args.eval_checkpoint:
            model = YOLO(args.eval_checkpoint)
            logger.info(f"Loaded YOLO model from: {args.eval_checkpoint}")
        else:
            logger.info(f"Using default model: {args.model}")

        if args.bin_distance:
            # --- Distance-binned path: route through evaluate_coco() ---
            logger.info("Distance binning enabled — using evaluate_coco() for YOLO")

            # Pre-extract frames to disk for faster loading (avoids repeated video decode)
            if args.use_gmind:
                if args.isp_variant:
                    frame_cache_base = Path(args.checkpoint_dir) / "torchvision_datasets" / args.isp_variant
                else:
                    frame_cache_base = Path(args.checkpoint_dir) / "torchvision_datasets"
                val_dataset.extract_frames(frame_cache_base / "val")

            adapted_model = _YOLOTorchVisionAdapter(model)
            stats, eval_results, eval_coco_gt, eval_img_ids = evaluate_coco(
                adapted_model,
                val_loader,
                device,
                val_dataset=val_dataset,
                subset=args.eval_subset,
            )

            # Log overall results
            logger.info("=" * 70)
            logger.info("YOLO Evaluation Results (via evaluate_coco)")
            logger.info("=" * 70)
            if stats is not None:
                logger.info(f"AP50-95: {stats[0]:.4f}")
                logger.info(f"AP50:    {stats[1]:.4f}")
                logger.info(f"AP75:    {stats[2]:.4f}")

            # Run distance binning
            distance_binned = None
            if stats is not None and eval_coco_gt is not None:
                from Evaluation.analysis.analysis_utils import compute_distance_binned_metrics
                from Annotation.annotation_generation import parse_camera_intrinsics_from_calibration

                dist_cfg = config.get("distance_eval")
                if dist_cfg is None:
                    logger.error("--bin-distance requires 'distance_eval' section in config")
                else:
                    # sensor_calibration.txt lives in the project root
                    project_root = Path(__file__).resolve().parent.parent
                    calib_path = project_root / "sensor_calibration.txt"
                    camera_matrix, _ = parse_camera_intrinsics_from_calibration(
                        str(calib_path), camera_name=sensor,
                    )
                    if camera_matrix is None:
                        logger.error(f"Failed to parse camera intrinsics from {calib_path}")
                    else:
                        distance_binned = compute_distance_binned_metrics(
                            coco_gt=eval_coco_gt,
                            coco_results=eval_results,
                            camera_matrix=camera_matrix,
                            camera_height=dist_cfg["camera_height"],
                            camera_pitch_deg=dist_cfg["camera_pitch_deg"],
                            bin_edges=dist_cfg["bins"],
                            evaluated_img_ids=eval_img_ids,
                        )
            else:
                logger.error(
                    "Cannot run distance binning: evaluation failed "
                    f"(stats={stats is not None}, coco_gt={eval_coco_gt is not None})"
                )

            # Save results
            if args.eval_output:
                results_dict = {
                    "model": args.model,
                    "checkpoint": args.eval_checkpoint,
                    "map50-95": float(stats[0]) if stats is not None else None,
                    "map50": float(stats[1]) if stats is not None else None,
                    "map75": float(stats[2]) if stats is not None else None,
                }
                if distance_binned is not None:
                    results_dict["distance_binned_metrics"] = distance_binned
                output_path = Path(args.eval_output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "w") as f:
                    json.dump(results_dict, f, indent=2)
                logger.info(f"Results saved to: {output_path}")
        else:
            # --- Standard YOLO path: model.val() (unchanged) ---
            # Resolve data.yaml path
            if args.isp_variant:
                yolo_export_dir = Path(args.checkpoint_dir) / "yolo_datasets" / args.isp_variant
            else:
                yolo_export_dir = checkpoint_dir / "yolo_dataset"
            data_yaml = yolo_export_dir / "data.yaml"

            # Export dataset if not already done
            if not data_yaml.exists():
                yolo_export_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Exporting GMIND dataset to YOLO format: {yolo_export_dir}")
                from DataLoader.export_to_yolo import export_gmind_to_yolo

                data_yaml = export_gmind_to_yolo(
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    out_dir=str(yolo_export_dir),
                )

            # Run validation
            device_str = str(device) if device.type == "cuda" else "cpu"
            metrics = model.val(data=str(data_yaml), device=device_str, split="val")

            # Log results
            logger.info("=" * 70)
            logger.info("YOLO Evaluation Results")
            logger.info("=" * 70)
            if hasattr(metrics, "box"):
                logger.info(f"AP50-95: {metrics.box.map:.4f}")
                logger.info(f"AP50:    {metrics.box.map50:.4f}")
                logger.info(f"AP75:    {metrics.box.map75:.4f}")

            # Save results if requested
            if args.eval_output:
                results_dict = {
                    "model": args.model,
                    "checkpoint": args.eval_checkpoint,
                    "map50-95": metrics.box.map,
                    "map50": metrics.box.map50,
                    "map75": metrics.box.map75,
                    "per_class_map50": metrics.box.maps.tolist()
                    if hasattr(metrics.box.maps, "tolist")
                    else list(metrics.box.maps),
                }
                output_path = Path(args.eval_output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "w") as f:
                    json.dump(results_dict, f, indent=2)
                logger.info(f"Results saved to: {output_path}")

        logger.info("Evaluation complete")
        return

    if args.eval_only:
        logger.info("Running in evaluation-only mode")
        if args.eval_checkpoint:
            if args.backend == "torchvision":
                load_model_weights(model, args.eval_checkpoint, device)
            else:
                from DeepLearning.adapters import get_model as adapter_get_model

                adapter_kwargs = {"pretrained": False}
                if args.eval_checkpoint:
                    adapter_kwargs["weights"] = args.eval_checkpoint
                model = adapter_get_model(
                    args.model, args.num_classes, backend=args.backend, **adapter_kwargs
                )
                model.to(device)

        # Create val_loader if not already created (for GMIND it's created above)
        if not args.use_gmind:
            val_loader = DataLoader(
                val_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=collate_fn,
            )

        # Extract frames to cache (avoids on-the-fly video decode)
        if args.use_gmind:
            if args.isp_variant:
                frame_cache_base = Path(args.checkpoint_dir) / "torchvision_datasets" / args.isp_variant
            else:
                frame_cache_base = Path(args.checkpoint_dir) / "torchvision_datasets"
            val_dataset.extract_frames(frame_cache_base / "val")

        detections_path = None
        if args.eval_output:
            eval_output_path = Path(args.eval_output)
            detections_path = eval_output_path.parent / f"{eval_output_path.stem}_detections.json"

        stats, eval_results, eval_coco_gt, eval_img_ids = evaluate_coco(
            model,
            val_loader,
            device,
            val_dataset=val_dataset,
            save_results=str(detections_path) if detections_path else None,
            subset=args.eval_subset,
        )
        if stats is not None:
            logger.info(f"COCO AP (0.5:0.95): {stats[0]:.4f}")

        # Distance-binned evaluation (opt-in)
        distance_binned = None
        if args.bin_distance:
            if stats is None or eval_coco_gt is None:
                logger.error(
                    "Cannot run distance binning: evaluation failed "
                    f"(stats={stats is not None}, coco_gt={eval_coco_gt is not None})"
                )
            else:
                from Evaluation.analysis.analysis_utils import compute_distance_binned_metrics
                from Annotation.annotation_generation import parse_camera_intrinsics_from_calibration

                dist_cfg = config.get("distance_eval")
                if dist_cfg is None:
                    logger.error("--bin-distance requires 'distance_eval' section in config")
                else:
                    # sensor_calibration.txt lives in the project root
                    project_root = Path(__file__).resolve().parent.parent
                    calib_path = project_root / "sensor_calibration.txt"
                    camera_matrix, _ = parse_camera_intrinsics_from_calibration(
                        str(calib_path), camera_name=sensor,
                    )
                    if camera_matrix is None:
                        logger.error(f"Failed to parse camera intrinsics from {calib_path}")
                    else:
                        distance_binned = compute_distance_binned_metrics(
                            coco_gt=eval_coco_gt,
                            coco_results=eval_results,
                            camera_matrix=camera_matrix,
                            camera_height=dist_cfg["camera_height"],
                            camera_pitch_deg=dist_cfg["camera_pitch_deg"],
                            bin_edges=dist_cfg["bins"],
                            evaluated_img_ids=eval_img_ids,
                        )

        # Save YOLO-style summary (model, checkpoint, mAP, distance bins)
        if args.eval_output:
            results_dict = {
                "model": args.model,
                "checkpoint": args.eval_checkpoint,
                "map50-95": float(stats[0]) if stats is not None else None,
                "map50": float(stats[1]) if stats is not None else None,
                "map75": float(stats[2]) if stats is not None else None,
            }
            if distance_binned is not None:
                results_dict["distance_binned_metrics"] = distance_binned
            output_path = Path(args.eval_output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(results_dict, f, indent=2)
            logger.info(f"Results saved to: {output_path}")

        logger.info("Evaluation complete")
        return
    # Handle Ultralytics models differently
    if is_ultralytics and args.use_gmind:
        # Ultralytics training path for GMIND dataset
        logger.info("=" * 60)
        logger.info("Ultralytics YOLO Training Path")
        logger.info("=" * 60)

        # Export GMIND dataset to YOLO format (shared across models for same variant)
        if args.isp_variant:
            yolo_export_dir = Path(args.checkpoint_dir) / "yolo_datasets" / args.isp_variant
        else:
            yolo_export_dir = checkpoint_dir / "yolo_dataset"
        data_yaml = yolo_export_dir / "data.yaml"

        # Get expected dataset sizes
        expected_train_size = len(train_dataset)
        expected_val_size = len(val_dataset)

        # Also get test set size for logging
        test_config = config.get("test", {})
        test_sets = test_config.get("sets", [])
        test_set_subdirs = test_config.get("set_subdirs", {})
        test_percentage_split = test_config.get("percentage_split", {})
        test_percentage_split_start = test_config.get("percentage_split_start", {})
        test_loader = get_gmind_dataloader(
            data_root=data_root,
            sets=test_sets,
            sensor=sensor,
            transforms=get_transform(train=False),
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            frame_stride=frame_stride,
            max_frames=config["data"].get("max_frames_per_video"),
            set_subdirs=test_set_subdirs,
            percentage_split=test_percentage_split,
            percentage_split_start=test_percentage_split_start,
            **isp_kwargs,
        )
        expected_test_size = len(test_loader.dataset)
        total_for_split = expected_train_size + expected_test_size

        logger.info("=" * 70)
        logger.info("Dataset Split Summary")
        logger.info("=" * 70)
        logger.info(
            f"  Train: {expected_train_size} frames ({expected_train_size/total_for_split*100:.1f}%)"
        )
        logger.info(
            f"  Validation: {expected_val_size} frames (same as train - for Ultralytics monitoring)"
        )
        logger.info(
            f"  Test: {expected_test_size} frames ({expected_test_size/total_for_split*100:.1f}%) - kept separate for final evaluation"
        )
        logger.info(f"  Total (train+test): {total_for_split} frames")
        logger.info(
            f"  Split: {expected_train_size/total_for_split*100:.1f}% train / {expected_test_size/total_for_split*100:.1f}% test"
        )
        logger.info("=" * 70)

        # Check if export already exists and is complete
        export_exists = (
            data_yaml.exists()
            and (yolo_export_dir / "train" / "images").exists()
            and (yolo_export_dir / "val" / "images").exists()
        )

        if export_exists:
            train_count = len(list((yolo_export_dir / "train" / "images").glob("*.jpg")))
            val_count = len(list((yolo_export_dir / "val" / "images").glob("*.jpg")))
            train_labels_count = len(list((yolo_export_dir / "train" / "labels").glob("*.txt")))
            val_labels_count = len(list((yolo_export_dir / "val" / "labels").glob("*.txt")))

            # Check if export is complete and matches expected sizes
            train_complete = (
                train_count == expected_train_size and train_labels_count == expected_train_size
            )
            val_complete = val_count == expected_val_size and val_labels_count == expected_val_size

            if train_complete and val_complete:
                logger.info(
                    f"YOLO format dataset already exists and is complete at {yolo_export_dir}"
                )
                logger.info(f"  Training images: {train_count}, Validation images: {val_count}")
                logger.info(f"  Using existing export: {data_yaml}")
            else:
                # Partial export detected
                logger.warning(f"Partial export detected at {yolo_export_dir}")
                logger.warning(f"  Expected: train={expected_train_size}, val={expected_val_size}")
                logger.warning(
                    f"  Found: train={train_count} (labels={train_labels_count}), val={val_count} (labels={val_labels_count})"
                )
                logger.info("Deleting incomplete export and re-exporting...")
                if yolo_export_dir.exists():
                    shutil.rmtree(yolo_export_dir)
                export_exists = False

        if not export_exists:
            yolo_export_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Exporting GMIND dataset to YOLO format: {yolo_export_dir}")
            logger.info(
                f"  Expected: train={expected_train_size} images, val={expected_val_size} images"
            )
            from DataLoader.export_to_yolo import export_gmind_to_yolo

            try:
                data_yaml = export_gmind_to_yolo(
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    out_dir=str(yolo_export_dir),
                )
                logger.info(f"Dataset exported successfully: {data_yaml}")
            except Exception as e:
                logger.error(f"Failed to export dataset to YOLO format: {e}")
                raise

        # Ensure data_yaml is a string path for Ultralytics
        data_yaml = str(data_yaml)

        # Setup TensorBoard
        tb_writer = None
        tb_log_dir = checkpoint_dir / "tensorboard_logs"
        if TENSORBOARD_AVAILABLE:
            tb_writer = SummaryWriter(log_dir=str(tb_log_dir))
            logger.info(f"TensorBoard logging enabled: {tb_log_dir}")
            logger.info(f"  View with: tensorboard --logdir={tb_log_dir}")
        else:
            logger.warning(
                "TensorBoard not available - training metrics will not be logged to TensorBoard"
            )
            logger.warning("  Install with: pip install tensorboard")

        # Configure Ultralytics training parameters
        train_kwargs = {
            "data": data_yaml,
            "epochs": args.epochs,
            "imgsz": 640,  # Standard YOLO input size
            "batch": args.batch_size,
            "device": str(device) if device.type == "cuda" else "cpu",
            "project": str(checkpoint_dir.resolve()),
            "name": "training",
            "exist_ok": True,
            "save": True,
            "save_period": 10,  # Save checkpoint every 10 epochs
        }

        # Enable TensorBoard logging in Ultralytics (it has built-in support)
        if TENSORBOARD_AVAILABLE:
            train_kwargs["plots"] = True  # Enable plots/metrics visualization
            # Ultralytics will automatically log to its runs directory
            # We'll also log to our custom TensorBoard directory

        # Add augmentation settings (Ultralytics has built-in augmentations)
        if args.augment_level == "light":
            aug_config = {
                "hsv_h": 0.01,  # Hue augmentation
                "hsv_s": 0.5,  # Saturation augmentation
                "hsv_v": 0.3,  # Value (brightness) augmentation
                "degrees": 5.0,  # Rotation augmentation
                "translate": 0.05,  # Translation augmentation
                "scale": 0.3,  # Scaling augmentation
                "flipud": 0.0,  # Vertical flip (disabled)
                "fliplr": 0.5,  # Horizontal flip
                "mosaic": 0.5,  # Mosaic augmentation
                "mixup": 0.0,  # Mixup augmentation (disabled)
                "copy_paste": 0.0,  # Copy-paste augmentation (disabled)
            }
        elif args.augment_level == "heavy":
            aug_config = {
                "hsv_h": 0.02,  # Hue augmentation
                "hsv_s": 0.9,  # Saturation augmentation
                "hsv_v": 0.5,  # Value (brightness) augmentation
                "degrees": 15.0,  # Rotation augmentation
                "translate": 0.15,  # Translation augmentation
                "scale": 0.7,  # Scaling augmentation
                "flipud": 0.1,  # Vertical flip
                "fliplr": 0.5,  # Horizontal flip
                "mosaic": 1.0,  # Mosaic augmentation
                "mixup": 0.2,  # Mixup augmentation
                "copy_paste": 0.1,  # Copy-paste augmentation
            }
        else:  # medium (default)
            aug_config = {
                "hsv_h": 0.015,  # Hue augmentation
                "hsv_s": 0.7,  # Saturation augmentation
                "hsv_v": 0.4,  # Value (brightness) augmentation
                "degrees": 10.0,  # Rotation augmentation
                "translate": 0.1,  # Translation augmentation
                "scale": 0.5,  # Scaling augmentation
                "flipud": 0.0,  # Vertical flip (disabled by default)
                "fliplr": 0.5,  # Horizontal flip
                "mosaic": 1.0,  # Mosaic augmentation
                "mixup": 0.1,  # Mixup augmentation
                "copy_paste": 0.0,  # Copy-paste augmentation (disabled)
            }

        train_kwargs.update(aug_config)
        logger.info(f"Using '{args.augment_level}' augmentation level")

        # RT-DETR specific settings: disable AMP (known to cause NaN with
        # transformer decoders) and use explicit optimizer instead of auto
        if "rtdetr" in args.model.lower():
            train_kwargs["amp"] = False
            train_kwargs["deterministic"] = False  # F.grid_sample incompatibility
            logger.info("RT-DETR detected: disabled AMP and deterministic mode (known NaN issues)")

        # Learning rate
        if args.lr:
            train_kwargs["lr0"] = args.lr

        # Resume from checkpoint if specified
        if args.resume:
            logger.info(f"Resuming from checkpoint: {args.resume}")
            train_kwargs["resume"] = args.resume

        # Log training configuration summary
        logger.info("=" * 70)
        logger.info("ULTRAlytics Training Configuration")
        logger.info("=" * 70)
        logger.info(f"Model: {args.model}")
        logger.info(f"Dataset: {expected_train_size} train, {expected_val_size} val images")
        logger.info(f"Epochs: {args.epochs}")
        logger.info(f"Batch size: {args.batch_size}")
        logger.info(f"Image size: 640x640")
        logger.info(f"Device: {device}")
        logger.info(f"Augmentation level: {args.augment_level}")
        if args.lr:
            logger.info(f"Learning rate: {args.lr}")
        logger.info(f"Checkpoint directory: {checkpoint_dir}")
        logger.info(f"TensorBoard logs: {tb_log_dir}")
        logger.info("=" * 70)
        logger.info("Starting Ultralytics training...")
        logger.info("Note: Ultralytics will show progress bars and metrics during training")
        logger.info("=" * 70)

        # Start training
        try:
            results = model.train(**train_kwargs)

            # Log training results summary
            logger.info("=" * 70)
            logger.info("Training Completed!")
            logger.info("=" * 70)

            # Extract and log final metrics
            if hasattr(results, "results_dict"):
                metrics = results.results_dict
                logger.info("Final Training Metrics:")
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        logger.info(f"  {key}: {value:.4f}")
                        if tb_writer:
                            tb_writer.add_scalar(f"train/{key}", value, args.epochs)
            elif hasattr(results, "results"):
                # Alternative results format
                logger.info(f"Training results available in: {results}")

            best_model_path = checkpoint_dir / "training" / "weights" / "best.pt"
            last_model_path = checkpoint_dir / "training" / "weights" / "last.pt"

            logger.info(f"Best model: {best_model_path}")
            logger.info(f"Last checkpoint: {last_model_path}")
            logger.info(f"TensorBoard logs: {tb_log_dir}")
            logger.info("=" * 70)

        except Exception as e:
            logger.error(f"Ultralytics training failed: {e}")
            raise
        finally:
            if tb_writer:
                tb_writer.close()

    else:
        # Standard PyTorch training path (for torchvision/MMDetection models)

        # Pre-extract frames to disk for faster loading (avoids repeated video open/seek/close)
        if args.use_gmind:
            if args.isp_variant:
                frame_cache_base = Path(args.checkpoint_dir) / "torchvision_datasets" / args.isp_variant
            else:
                frame_cache_base = checkpoint_dir / "torchvision_datasets"

            logger.info("Pre-extracting frames to disk for faster training...")
            train_dataset.extract_frames(frame_cache_base / "train")
            val_dataset.extract_frames(frame_cache_base / "val")

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(trainable_params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
        logger.info(f"Using batch size: {args.batch_size}")

        # Create dataloaders (already created for GMIND, create for COCO)
        if args.use_gmind:
            # Dataloaders already created above
            pass
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                collate_fn=collate_fn,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=collate_fn,
            )

        scaler = torch.cuda.amp.GradScaler() if args.use_amp and device.type == "cuda" else None
        start_epoch = 0
        if args.resume:
            logger.info(f"Resuming from checkpoint: {args.resume}")
            start_epoch, scaler_state = load_checkpoint(model, optimizer, args.resume, device)
            if scaler and scaler_state is not None:
                scaler.load_state_dict(scaler_state)

        # Setup TensorBoard
        tb_writer = None
        if TENSORBOARD_AVAILABLE:
            tb_log_dir = checkpoint_dir / "tensorboard_logs"
            tb_writer = SummaryWriter(log_dir=str(tb_log_dir))
            logger.info(f"TensorBoard logging enabled: {tb_log_dir}")

        best_map = -1.0
        best_epoch = -1

        try:
            logger.info(f"Starting training for {args.epochs} epochs")
            for epoch in range(start_epoch, args.epochs):
                start_time = time.time()
                train_loss = train_one_epoch(
                    model, optimizer, train_loader, device, epoch, scaler=scaler
                )
                lr_scheduler.step()
                epoch_time = time.time() - start_time
                current_lr = optimizer.param_groups[0]["lr"]

                # Log to TensorBoard
                if tb_writer:
                    tb_writer.add_scalar("train/loss", train_loss, epoch)
                    tb_writer.add_scalar("train/learning_rate", current_lr, epoch)
                    tb_writer.add_scalar("train/epoch_time", epoch_time, epoch)

                logger.info(
                    f"Epoch {epoch}/{args.epochs-1} | Loss: {train_loss:.4f} | LR: {current_lr:.6f} | Time: {epoch_time:.1f}s"
                )

                # Evaluate before saving so mAP is available for best-model tracking
                current_map = None
                if args.do_eval:
                    stats, _, _, _ = evaluate_coco(model, val_loader, device, val_dataset=val_dataset)
                    if stats is not None:
                        current_map = stats[0]
                        logger.info(f"COCO AP (0.5:0.95): {current_map:.4f}")
                        if tb_writer:
                            tb_writer.add_scalar("val/mAP", current_map, epoch)
                else:
                    model.eval()
                    with torch.no_grad():
                        val_images, val_targets = next(iter(val_loader))
                        val_images = [img.to(device) for img in val_images]
                        outputs = model(val_images)
                        num_dets = len(outputs[0].get("boxes", []))
                        logger.info(f"Sample validation: {num_dets} detections")

                # Save checkpoints: always last.pth, conditionally best.pth
                checkpoint = {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scaler_state": scaler.state_dict() if scaler is not None else None,
                    "args": vars(args),
                }
                save_checkpoint(checkpoint, str(checkpoint_dir), "last.pth")

                if current_map is not None and current_map > best_map:
                    best_map = current_map
                    best_epoch = epoch
                    save_checkpoint(checkpoint, str(checkpoint_dir), "best.pth")
                    logger.info(f"New best model at epoch {epoch} with mAP {best_map:.4f}")

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        finally:
            if best_epoch >= 0:
                logger.info(f"Best model from epoch {best_epoch} with mAP {best_map:.4f}")
            # Shutdown DataLoader workers
            if hasattr(train_loader, '_workers'):
                train_loader._shutdown_workers()
            if hasattr(val_loader, '_workers'):
                val_loader._shutdown_workers()
            # Release GPU memory
            del model
            torch.cuda.empty_cache()
            # Close TensorBoard writer
            if tb_writer:
                tb_writer.close()
            logger.info("Cleanup complete")


if __name__ == "__main__":
    main()
