"""MMDetection adapter for object detection models.

Provides a unified interface to build, train, and infer with MMDetection models.
Requires MMDetection to be installed: https://github.com/open-mmlab/mmdetection
"""

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Mapping of model short names to MMDetection config patterns
MODEL_CONFIG_MAPPING = {
    # Two-stage detectors
    "faster_rcnn": "faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py",
    "cascade_rcnn": "cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py",
    "mask_rcnn": "mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py",
    "hybrid_task_cascade": "htc/htc_r50_fpn_1x_coco.py",
    "libra_rcnn": "libra_rcnn/libra_rcnn_r50_fpn_1x_coco.py",
    # One-stage detectors
    "ssd": "ssd/ssd300_coco.py",
    "retinanet": "retinanet/retinanet_r50_fpn_1x_coco.py",
    "fcos": "fcos/fcos_r50_fpn_1x_coco.py",
    "atss": "atss/atss_r50_fpn_1x_coco.py",
    "gfl": "gfl/gfl_r50_fpn_1x_coco.py",
    "vfnet": "vfnet/vfnet_r50_fpn_1x_coco.py",
    # YOLO variants
    "yolov3": "yolo/yolov3_d53_320_273e_coco.py",
    "yolov4": "yolov4/yolov4_p5_mstrain_syncbn_300e_coco.py",
    "yolov5": "yolox/yolov5_l_300e_coco.py",
    "yolov6": "yolov6/yolov6_m_300e_coco.py",
    "yolov7": "yolov7/yolov7_x_syncbn_fast_8x16b_300e_coco.py",
    # Transformer-based
    "detr": "detr/detr_r50_8x2_150e_coco.py",
    "dino": "dino/dino-5scale_swin-l_8xb2-12e_coco.py",
    "conditional_detr": "conditional_detr/conditional_detr_r50_8x2_150e_coco.py",
    "deformable_detr": "deformable_detr/deformable_detr_r50_16x2_50e_coco.py",
    # Others
    "reppoints": "reppoints/reppoints_moment_r50_fpn_1x_coco.py",
    "foveabox": "foveabox/fovea_r50_fpn_4x4_1x_coco.py",
    "centernet": "centernet/centernet_hg104_4x16_16e_coco.py",
    "cornerdet": "cornerdet/corner_detect_hg104_4x16_16e_coco.py",
    "efficientdet": "efficientdet/efficientdet-d0_8x4_300e_coco.py",
}


def is_available() -> bool:
    """Check if MMDetection is installed."""
    try:
        import mmcv  # type: ignore
        import mmdet  # type: ignore

        return True
    except ImportError:
        return False
    except Exception:
        # Other exceptions might indicate partial installation
        return False


def find_config_file(model_name: str, mmdet_dir: Optional[str] = None) -> str:
    """Locate MMDetection config file for a model.

    Args:
        model_name: Model identifier or path to config file
        mmdet_dir: Optional path to MMDetection configs directory

    Returns:
        Path to config file

    Raises:
        FileNotFoundError: If config cannot be found
    """
    # If it's already a file path, return it
    if os.path.isfile(model_name):
        return model_name

    # Try to find in MMDetection installation
    if not mmdet_dir:
        try:
            import mmdet

            mmdet_path = os.path.dirname(mmdet.__file__)
            # MMDetection configs are typically in a separate configs directory
            # Check common locations
            possible_paths = [
                os.path.join(mmdet_path, "..", "configs"),
                os.path.join(mmdet_path, "configs"),
                os.path.join(os.path.dirname(mmdet_path), "configs"),
            ]
            for path in possible_paths:
                abs_path = os.path.abspath(path)
                if os.path.isdir(abs_path):
                    mmdet_dir = abs_path
                    break
        except Exception:
            mmdet_dir = None

    # Look up in mapping
    if model_name in MODEL_CONFIG_MAPPING:
        config_relative = MODEL_CONFIG_MAPPING[model_name]
        if mmdet_dir:
            config_path = os.path.join(mmdet_dir, config_relative)
            if os.path.isfile(config_path):
                return config_path

    # Try to find in standard MMDetection location
    if mmdet_dir and os.path.isdir(mmdet_dir):
        config_path = os.path.join(mmdet_dir, model_name + ".py")
        if os.path.isfile(config_path):
            return config_path

    raise FileNotFoundError(
        f"Cannot find config for model '{model_name}'. "
        f"Provide explicit config_file path or ensure MMDetection configs are available. "
        f"MMDetection configs can be downloaded from: "
        f"https://github.com/open-mmlab/mmdetection/tree/main/configs"
    )


def get_model(
    model_name: str,
    num_classes: int = 80,
    pretrained: bool = True,
    config_file: Optional[str] = None,
    weights: Optional[str] = None,
    checkpoint: Optional[str] = None,
    **kwargs,
) -> Any:
    """Build an MMDetection model.

    Args:
        model_name: Model identifier or path to config file
        num_classes: Number of classes for the dataset
        pretrained: Whether to load pretrained weights
        config_file: Explicit path to MMDetection config file (.py)
        weights: Path to checkpoint file to load
        checkpoint: Alias for weights
        **kwargs: Additional arguments (ignored for MMDetection)

    Returns:
        Built MMDetection model ready for training or inference

    Raises:
        RuntimeError: If MMDetection not installed or model build fails
    """
    if not is_available():
        raise RuntimeError(
            "MMDetection is not installed.\n"
            "Install it with: pip install mmdet mmcv\n"
            "Or see: https://github.com/open-mmlab/mmdetection"
        )

    try:
        # Try mmengine first (newer MMDetection versions)
        try:
            from mmengine import Config  # type: ignore
        except ImportError:
            # Fallback to mmcv for older versions
            from mmcv import Config  # type: ignore
        # Note: build_detector is not available in MMDetection 3.x, we use mmengine.registry instead
    except ImportError as e:
        raise RuntimeError(
            f"Failed to import MMDetection components: {e}\n"
            "Make sure both mmdet and mmcv are installed:\n"
            "  pip install mmdet mmcv\n"
            "For MMDetection 3.x, you may also need mmengine:\n"
            "  pip install mmengine"
        )

    # Resolve config file path
    try:
        cfg_path = config_file or find_config_file(model_name)
    except FileNotFoundError as e:
        raise RuntimeError(
            f"Config not found: {e}\n"
            "You can either:\n"
            "  1. Provide explicit config_file path\n"
            "  2. Clone MMDetection repo to get configs: "
            "git clone https://github.com/open-mmlab/mmdetection.git\n"
            "  3. Download specific config files from the MMDetection repo"
        )

    logger.info(f"Loading config from: {cfg_path}")

    # Load config
    try:
        cfg = Config.fromfile(cfg_path)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load config from '{cfg_path}': {e}\n"
            "Make sure the config file is valid and all dependencies are installed."
        )

    # Optionally update num_classes in config
    try:
        if "roi_head" in cfg.model:
            roi_head = cfg.model.roi_head
            if hasattr(roi_head, "bbox_head"):
                bbox_head = roi_head.bbox_head
                if isinstance(bbox_head, list):
                    for bh in bbox_head:
                        if hasattr(bh, "num_classes"):
                            bh.num_classes = num_classes
                            logger.info(f"Set num_classes={num_classes} in bbox_head")
                elif hasattr(bbox_head, "num_classes"):
                    bbox_head.num_classes = num_classes
                    logger.info(f"Set num_classes={num_classes} in bbox_head")
        # Also check for single-stage detectors
        if "bbox_head" in cfg.model:
            bbox_head = cfg.model.bbox_head
            if hasattr(bbox_head, "num_classes"):
                bbox_head.num_classes = num_classes
                logger.info(f"Set num_classes={num_classes} in bbox_head")
    except Exception as e:
        logger.warning(f"Could not set num_classes in config: {e}")

    # Build model using MMEngine registry (MMDetection 3.x)
    try:
        from mmengine.registry import MODELS, build_from_cfg

        model = build_from_cfg(cfg.model, MODELS)
        logger.info(f"Successfully built MMDetection model from config")
    except ModuleNotFoundError as e:
        if "mmcv._ext" in str(e):
            raise RuntimeError(
                f"MMDetection requires mmcv with compiled extensions.\n"
                f"Error: {e}\n\n"
                "To fix this, install mmcv with compiled extensions:\n"
                "  1. Install openmim: pip install openmim\n"
                "  2. Install mmcv: mim install mmcv\n"
                "  Or use a pre-built wheel for your CUDA/PyTorch version.\n"
                "  See: https://mmcv.readthedocs.io/en/latest/get_started/installation.html"
            )
        raise
    except Exception as e:
        raise RuntimeError(
            f"Failed to build MMDetection model: {e}\n"
            "This might be due to:\n"
            "  - Missing dependencies in the config\n"
            "  - Incompatible MMDetection version\n"
            "  - Invalid config file\n"
            "  - Missing mmcv compiled extensions (mmcv._ext)"
        )

    # Load checkpoint if provided
    if weights or checkpoint:
        checkpoint_path = weights or checkpoint
        if os.path.isfile(checkpoint_path):
            try:
                # Use mmengine checkpoint loading (works for both MMDetection 2.x and 3.x)
                from mmengine.runner import CheckpointLoader  # type: ignore

                checkpoint_data = CheckpointLoader.load_checkpoint(
                    checkpoint_path, map_location="cpu"
                )
                if "state_dict" in checkpoint_data:
                    model.load_state_dict(checkpoint_data["state_dict"], strict=False)
                else:
                    model.load_state_dict(checkpoint_data, strict=False)
                logger.info(f"Loaded checkpoint from: {checkpoint_path}")
            except ImportError:
                # Fallback to torch.load if mmengine not available
                import torch

                checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
                if "state_dict" in checkpoint_data:
                    model.load_state_dict(checkpoint_data["state_dict"], strict=False)
                else:
                    model.load_state_dict(checkpoint_data, strict=False)
                logger.info(f"Loaded checkpoint from: {checkpoint_path}")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
                import traceback

                traceback.print_exc()
    elif pretrained:
        # Try to load pretrained weights from config
        if "load_from" in cfg:
            try:
                from mmengine.runner import CheckpointLoader  # type: ignore

                checkpoint_data = CheckpointLoader.load_checkpoint(
                    cfg.load_from, map_location="cpu"
                )
                if "state_dict" in checkpoint_data:
                    model.load_state_dict(checkpoint_data["state_dict"], strict=False)
                else:
                    model.load_state_dict(checkpoint_data, strict=False)
                logger.info(f"Loaded pretrained weights from config")
            except Exception as e:
                logger.debug(f"Could not load pretrained weights: {e}")
        elif "pretrained" in cfg.model and cfg.model.pretrained:
            try:
                from mmengine.runner import CheckpointLoader  # type: ignore

                checkpoint_data = CheckpointLoader.load_checkpoint(
                    cfg.model.pretrained, map_location="cpu"
                )
                if "state_dict" in checkpoint_data:
                    model.load_state_dict(checkpoint_data["state_dict"], strict=False)
                else:
                    model.load_state_dict(checkpoint_data, strict=False)
                logger.info(f"Loaded pretrained weights from config")
            except Exception as e:
                logger.debug(f"Could not load pretrained weights: {e}")

    return model


def inference(model: Any, image_path: str, **kwargs) -> Dict[str, Any]:
    """Run inference with an MMDetection model.

    Args:
        model: MMDetection model object
        image_path: Path to input image
        **kwargs: Additional arguments

    Returns:
        Detection results dictionary
    """
    if not is_available():
        raise RuntimeError("MMDetection is not installed")

    try:
        from mmdet.apis import inference_detector  # type: ignore

        result = inference_detector(model, image_path)
        logger.info(f"Inference completed on: {image_path}")
        return result
    except Exception as e:
        raise RuntimeError(f"MMDetection inference failed: {e}")
