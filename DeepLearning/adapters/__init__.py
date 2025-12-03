"""Unified model loading interface for all supported detection backends.

This module provides a single entry point (get_model) that automatically detects
and delegates to the appropriate adapter (TorchVision, Ultralytics, MMDetection).

Each backend is implemented as a separate adapter module with consistent interface:
- is_available(): Check if the library is installed
- get_model(): Build and return a model
- inference(): Run inference (optional)

Usage:
    from DeepLearning.adapters import get_model

    # Auto-detect backend from model name
    model = get_model('yolov8m', num_classes=80)

    # Or explicit backend
    model = get_model('fasterrcnn_resnet50_fpn', num_classes=80, backend='torchvision')
"""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Import adapter modules
from . import mmdet_adapter, torchvision_adapter, ultralytics_adapter


def detect_backend(model_name: str) -> str:
    """Detect which backend a model belongs to based on its name.

    Args:
        model_name: Model identifier string

    Returns:
        Backend name: 'torchvision', 'ultralytics', or 'mmdetection'
    """
    model_lower = model_name.lower()

    # Explicit patterns that must be handled first
    if model_lower.startswith("yolox"):
        return "mmdetection"
    if model_lower.startswith("detr_"):
        return "torchvision"

    # Ultralytics YOLO patterns (distinctive)
    if any(x in model_lower for x in ["yolov", "rtdetr"]):
        return "ultralytics"

    # TorchVision patterns (check before MMDetection since there's overlap)
    torchvision_models = [
        "fasterrcnn_resnet",
        "fasterrcnn_mobilenet",
        "maskrcnn_resnet",
        "maskrcnn_mobilenet",
        "retinanet_resnet",
        "ssd300_vgg",
        "ssdlite320",
        "fcos_resnet",
        "keypointrcnn",
    ]
    if any(x in model_lower for x in torchvision_models):
        return "torchvision"

    # MMDetection patterns (after torchvision check to avoid conflicts)
    mmdet_models = [
        "faster_rcnn",
        "cascade_rcnn",
        "mask_rcnn",
        "hybrid_task_cascade",
        "libra_rcnn",
        "ssd",
        "retinanet",
        "fcos",
        "atss",
        "gfl",
        "vfnet",
        "yolov3",
        "yolov4",
        "yolov5",
        "yolov6",
        "yolov7",
        "detr",
        "deformable_detr",
        "dino",
        "conditional_detr",
        "efficientdet",
        "cornerdet",
        "centernet",
        "paa",
        "reppoints",
        "foveabox",
        "panoptic_fpn",
        "yolact",
        "solo",
        "blendmask",
    ]
    if any(x in model_lower for x in mmdet_models):
        return "mmdetection"

    # Default to torchvision
    return "torchvision"


def get_model(
    model_name: str, num_classes: int = 80, pretrained: bool = True, backend: str = "auto", **kwargs
) -> Any:
    """Load a detection model from any supported backend.

    Automatically detects the backend if not specified, then delegates to the
    appropriate adapter.

    Args:
        model_name: Model identifier (e.g., 'fasterrcnn_resnet50_fpn', 'yolov8m', 'faster_rcnn')
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
    backend: Backend to use ('auto' for automatic detection, or explicit:
        'torchvision', 'ultralytics', or 'mmdetection')
        **kwargs: Additional backend-specific arguments

    Returns:
        Loaded model ready for training or inference

    Raises:
        ValueError: If backend not available or model loading fails

    Examples:
        >>> # Auto-detect backend
        >>> model = get_model('yolov8m', num_classes=80)
        >>>
        >>> # Explicit backend
        >>> model = get_model('fasterrcnn_resnet50_fpn', num_classes=80, backend='torchvision')
        >>>
        >>> # With extra args (backend-specific)
        >>> model = get_model('faster_rcnn', num_classes=80, backend='mmdetection',
        ...                   config_file='/path/to/config.py')
    """
    # Auto-detect backend if needed
    if backend == "auto":
        backend = detect_backend(model_name)
        logger.debug(f"Auto-detected backend: {backend} for model: {model_name}")

    # Dispatch to appropriate adapter
    if backend == "torchvision":
        return _get_model_torchvision(model_name, num_classes, pretrained, **kwargs)
    elif backend in ("ultralytics", "yolo"):
        return _get_model_ultralytics(model_name, num_classes, pretrained, **kwargs)
    elif backend == "mmdetection":
        return _get_model_mmdetection(model_name, num_classes, pretrained, **kwargs)
    else:
        raise ValueError(
            f"Unknown backend: {backend}. "
            f"Choose from: 'torchvision', 'ultralytics', 'mmdetection', 'auto'"
        )


def _get_model_torchvision(model_name: str, num_classes: int, pretrained: bool, **kwargs) -> Any:
    """Load a TorchVision model via adapter."""
    if not torchvision_adapter.is_available():
        raise ValueError("TorchVision is not installed")

    try:
        model = torchvision_adapter.get_model(
            model_name=model_name, num_classes=num_classes, pretrained=pretrained, **kwargs
        )
        logger.info(f"Loaded TorchVision model: {model_name}")
        return model
    except Exception as e:
        logger.error(f"Failed to load TorchVision model {model_name}: {e}")
        raise


def _get_model_ultralytics(model_name: str, num_classes: int, pretrained: bool, **kwargs) -> Any:
    """Load an Ultralytics YOLO model via adapter."""
    if not ultralytics_adapter.is_available():
        raise ValueError("Ultralytics is not installed. Install with: pip install ultralytics")

    try:
        model = ultralytics_adapter.get_model(
            model_name=model_name, num_classes=num_classes, pretrained=pretrained, **kwargs
        )
        logger.info(f"Loaded Ultralytics model: {model_name}")
        return model
    except Exception as e:
        logger.error(f"Failed to load Ultralytics model {model_name}: {e}")
        raise


def _get_model_mmdetection(model_name: str, num_classes: int, pretrained: bool, **kwargs) -> Any:
    """Load an MMDetection model via adapter."""
    if not mmdet_adapter.is_available():
        raise ValueError(
            "MMDetection is not installed. Install following: "
            "pip install mmdet mmcv\n"
            "Or see: https://github.com/open-mmlab/mmdetection"
        )

    try:
        model = mmdet_adapter.get_model(
            model_name=model_name, num_classes=num_classes, pretrained=pretrained, **kwargs
        )
        logger.info(f"Loaded MMDetection model: {model_name}")
        return model
    except Exception as e:
        logger.error(f"Failed to load MMDetection model {model_name}: {e}")
        raise


__all__ = [
    "get_model",
    "detect_backend",
    "torchvision_adapter",
    "ultralytics_adapter",
    "mmdet_adapter",
]
