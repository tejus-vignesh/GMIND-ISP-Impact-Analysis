"""TorchVision adapter for object detection models.

Provides a unified interface to build and use TorchVision detection models.
"""

import logging
from typing import Any, Optional

import torch
import torchvision

logger = logging.getLogger(__name__)


def is_available() -> bool:
    """Check if TorchVision is installed."""
    try:
        import torchvision  # type: ignore

        return True
    except Exception:
        return False


def get_model(model_name: str, num_classes: int = 91, pretrained: bool = True, **kwargs) -> Any:
    """Build a TorchVision detection model.

    Args:
        model_name: Model identifier (e.g., 'fasterrcnn_resnet50_fpn', 'ssd300_vgg16')
        num_classes: Number of classes for the dataset
        pretrained: Whether to load pretrained weights
        **kwargs: Additional arguments (ignored)

    Returns:
        Built TorchVision model ready for training or inference

    Raises:
        ValueError: If model not available or building fails
    """
    if not is_available():
        raise ValueError("TorchVision is not installed")

    model_name = model_name.lower()
    weights = "DEFAULT" if pretrained else None

    try:
        # Two-stage: Faster R-CNN variants
        if model_name == "fasterrcnn_resnet50_fpn":
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                weights=weights, progress=True
            )
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = (
                torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
            )
            logger.info(f"Loaded model: {model_name} with {num_classes} classes")
            return model

        if model_name == "fasterrcnn_resnet50_fpn_v2":
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
                weights=weights, progress=True
            )
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = (
                torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
            )
            logger.info(f"Loaded model: {model_name} with {num_classes} classes")
            return model

        if model_name == "fasterrcnn_mobilenet_v3_large_fpn":
            model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
                weights=weights, progress=True
            )
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = (
                torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
            )
            logger.info(f"Loaded model: {model_name} with {num_classes} classes")
            return model

        if model_name == "fasterrcnn_mobilenet_v3_large_320_fpn":
            model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(
                weights=weights, progress=True
            )
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = (
                torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
            )
            logger.info(f"Loaded model: {model_name} with {num_classes} classes")
            return model

        # Two-stage: Mask R-CNN
        if model_name == "maskrcnn_resnet50_fpn":
            model = torchvision.models.detection.maskrcnn_resnet50_fpn(
                weights=weights, progress=True
            )
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = (
                torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
            )
            logger.info(f"Loaded model: {model_name} with {num_classes} classes")
            return model

        if model_name == "maskrcnn_resnet50_fpn_v2":
            model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
                weights=weights, progress=True
            )
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = (
                torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
            )
            logger.info(f"Loaded model: {model_name} with {num_classes} classes")
            return model

        # Single-stage: SSD variants
        if model_name == "ssd300_vgg16":
            model = torchvision.models.detection.ssd300_vgg16(weights=weights)
            try:
                model.head.classification_head.num_classes = num_classes
                logger.info(f"Loaded model: {model_name} with {num_classes} classes")
            except Exception as e:
                logger.warning(f"Could not set num_classes for ssd300_vgg16: {e}")
            return model

        if model_name == "ssdlite320_mobilenet_v3_large":
            model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=weights)
            try:
                if hasattr(model, "head") and hasattr(model.head, "classification_head"):
                    model.head.classification_head.num_classes = num_classes
                logger.info(f"Loaded model: {model_name} with {num_classes} classes")
            except Exception as e:
                logger.warning(f"Could not set num_classes for ssdlite320: {e}")
            return model

        # Single-stage: RetinaNet
        if model_name in ("retinanet_resnet50_fpn", "retinanet_resnet50_fpn_v2"):
            logger.warning(
                "RetinaNet with custom num_classes may cause shape mismatches. "
                "Consider using Faster R-CNN instead."
            )

            if model_name == "retinanet_resnet50_fpn_v2":
                model = torchvision.models.detection.retinanet_resnet50_fpn_v2(
                    weights=weights, progress=True
                )
            else:
                model = torchvision.models.detection.retinanet_resnet50_fpn(
                    weights=weights, progress=True
                )

            try:
                num_anchors = model.head.classification_head.num_anchors
                in_channels = model.head.classification_head.conv[0].in_channels
                model.head = torchvision.models.detection.retinanet.RetinaNetHead(
                    in_channels, num_anchors, num_classes
                )
                logger.info(f"Loaded model: {model_name} with {num_classes} classes")
            except Exception as e:
                logger.warning(f"Could not reconstruct RetinaNet head: {e}")
            return model

        # Anchor-free: FCOS
        if model_name == "fcos_resnet50_fpn":
            model = torchvision.models.detection.fcos_resnet50_fpn(weights=weights, progress=True)
            logger.info(f"Loaded model: {model_name}")
            return model

        # Keypoint: Keypoint R-CNN
        if model_name == "keypointrcnn_resnet50_fpn":
            model = torchvision.models.detection.keypointrcnn_resnet50_fpn(
                weights=weights, progress=True
            )
            logger.info(f"Loaded model: {model_name}")
            return model

    except AttributeError as e:
        raise ValueError(
            f"Model '{model_name}' not available in this torchvision build: {e}"
        ) from e
    except Exception as e:
        raise ValueError(f"Failed to load model '{model_name}': {e}") from e

    raise ValueError(
        f"Model '{model_name}' is not supported. "
        f"Available TorchVision models: fasterrcnn_resnet50_fpn, fasterrcnn_mobilenet_v3_large_fpn, "
        f"maskrcnn_resnet50_fpn, ssd300_vgg16, ssdlite320_mobilenet_v3_large, retinanet_resnet50_fpn, etc."
    )


def inference(model: Any, images: Any, **kwargs) -> Any:
    """Run inference with a TorchVision model.

    Args:
        model: TorchVision model object
        images: List of images or single image tensor
        **kwargs: Additional arguments (ignored)

    Returns:
        Detection results (list of dicts with 'boxes', 'labels', 'scores')
    """
    model.eval()
    with torch.no_grad():
        if isinstance(images, list):
            results = model(images)
        else:
            results = model([images])
    return results
