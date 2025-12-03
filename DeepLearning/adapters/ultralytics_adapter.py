"""Ultralytics YOLO adapter for object detection models.

Provides a unified interface to build, train, and infer with Ultralytics YOLO models.
Supports YOLOv5, YOLOv8, YOLOv9, YOLOv10, YOLOv11, YOLOv12, and RT-DETR models.

Requires Ultralytics to be installed: pip install ultralytics
"""

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def is_available() -> bool:
    """Check if Ultralytics YOLO is installed."""
    try:
        import ultralytics  # type: ignore

        return True
    except Exception:
        return False


def get_model(
    model_name: str,
    num_classes: int = 80,
    pretrained: bool = True,
    weights: Optional[str] = None,
    config: Optional[str] = None,
) -> Any:
    """Return an Ultralytics YOLO model object.

    Args:
        model_name: Can be:
            - a YOLO model identifier (e.g., 'yolov8m', 'yolov5l', 'yolov9c')
            - a path to a .pt weights file
            - a path to a YAML model config
        num_classes: Number of classes (used for model initialization)
        pretrained: Whether to load pretrained weights
        weights: Optional explicit path to weights
        config: Optional explicit path to config YAML

    Returns:
        Ultralytics YOLO model object ready for training or inference

    Raises:
        RuntimeError: If Ultralytics is not installed or model loading fails
    """
    if not is_available():
        raise RuntimeError(
            "Ultralytics (YOLO) package is not installed.\n"
            "Install it with: pip install ultralytics"
        )

    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as e:
        raise RuntimeError(f"Failed to import ultralytics.YOLO: {e}")

    # Determine source (weights or model name)
    source = None
    if weights and os.path.exists(weights):
        source = weights
        logger.info(f"Loading YOLO model from weights: {weights}")
    elif config and os.path.exists(config):
        source = config
        logger.info(f"Loading YOLO model from config: {config}")
    else:
        # Use model_name as identifier (e.g., 'yolov8m', 'yolov5l')
        source = model_name
        # Append .pt extension if not present (standard Ultralytics convention)
        if pretrained and not source.endswith((".pt", ".yaml")):
            source = source + ".pt" if not "/" in source else source
        logger.info(f"Loading YOLO model: {source}")

    try:
        model = YOLO(source)

        # Log model info
        logger.info(f"Successfully loaded model: {source}")
        logger.info(f"Model type: {model.task if hasattr(model, 'task') else 'unknown'}")

        return model
    except FileNotFoundError as e:
        # If model not found, try fallback for YOLOv12 -> YOLOv8 (most stable)
        if "yolov12" in model_name.lower():
            # Extract size suffix (n, s, m, l, x)
            size_suffix = model_name.lower().replace("yolov12", "")
            fallback_model = f"yolov8{size_suffix}"
            logger.warning(
                f"YOLOv12 model '{model_name}' not found. Trying fallback: {fallback_model}"
            )
            try:
                model = YOLO(fallback_model)
                logger.info(f"Successfully loaded fallback model: {fallback_model}")
                return model
            except Exception as e2:
                # Try yolov11 as second fallback
                fallback2 = f"yolov11{size_suffix}"
                logger.warning(
                    f"Fallback '{fallback_model}' also failed. Trying second fallback: {fallback2}"
                )
                try:
                    model = YOLO(fallback2)
                    logger.info(f"Successfully loaded second fallback model: {fallback2}")
                    return model
                except Exception as e3:
                    raise RuntimeError(
                        f"Failed to create Ultralytics YOLO model from '{source}' and fallbacks '{fallback_model}'/'{fallback2}': {e3}\n"
                        f"Original error: {e}\n"
                        f"Available models: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x"
                    )
        raise RuntimeError(f"Failed to create Ultralytics YOLO model from '{source}': {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to create Ultralytics YOLO model from '{source}': {e}")


def inference(model: Any, source: Any, conf: float = 0.25, **kwargs) -> Any:
    """Run inference with a YOLO model.

    Args:
        model: Ultralytics YOLO model object
        source: Image(s), folder, or video file
        conf: Confidence threshold
        **kwargs: Additional arguments passed to model.predict()

    Returns:
        Prediction results
    """
    if not is_available():
        raise RuntimeError("Ultralytics is not installed")

    try:
        results = model.predict(source=source, conf=conf, **kwargs)
        return results
    except Exception as e:
        raise RuntimeError(f"YOLO inference failed: {e}")


def export_coco_to_yolo(dataset, out_dir: str):
    """Export a COCO-style dataset (CocoDetectionWrapper) to Ultralytics YOLO format.

    - dataset: expected to be a CocoDetectionWrapper instance
    - out_dir: target directory where `images/` and `labels/` will be created and a `data.yaml` saved

    Returns the path to the generated data YAML.
    """
    import json
    from pathlib import Path

    p = Path(out_dir)
    images_dir = p / "images"
    labels_dir = p / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # build categories mapping
    cats = {}
    if hasattr(dataset, "coco"):
        coco = dataset.coco
        for c in coco.loadCats(coco.getCatIds()):
            cats[c["id"]] = c["name"]
    # iterate dataset and save images and txt labels
    img_paths = []
    for idx in range(len(dataset)):
        img, target = dataset[idx]
        # img is a tensor if transforms applied; handle PIL too
        from PIL import Image

        if hasattr(img, "numpy"):
            arr = img.mul(255).byte().permute(1, 2, 0).numpy()
            pil = Image.fromarray(arr)
        else:
            pil = img

        img_name = f"{idx:06d}.jpg"
        img_path = images_dir / img_name
        pil.save(img_path)
        img_paths.append(str(img_path))

        # write labels in YOLO format (class x_center y_center width height) normalized
        boxes = target.get("boxes")
        labels = target.get("labels")
        h, w = pil.size[1], pil.size[0]
        label_lines = []
        for b, l in zip(boxes, labels):
            x1, y1, x2, y2 = [float(x) for x in b]
            cx = (x1 + x2) / 2.0 / w
            cy = (y1 + y2) / 2.0 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            # YOLO expects 0-based class ids
            cls = int(l) - 1 if int(l) > 0 else int(l)
            label_lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        with open(labels_dir / (img_name.replace(".jpg", ".txt")), "w") as f:
            f.write("\n".join(label_lines))

    # create data yaml
    data_yaml = p / "data.yaml"
    data = {
        "train": str(images_dir),
        "val": str(images_dir),
        "names": [cats.get(k, str(k)) for k in sorted(cats.keys())] if cats else [],
    }
    with open(data_yaml, "w") as f:
        json.dump(data, f)

    return str(data_yaml)


def train_yolo(model, data_yaml: str, epochs: int = 10, imgsz: int = 640, **kwargs) -> Dict:
    """Train an Ultralytics YOLO model object using a generated data YAML.

    - model: ultralytics.YOLO instance
    - data_yaml: path to the data.yaml describing train/val and names
    - returns the training results dict
    """
    try:
        # ultralytics model exposes .train()
        res = model.train(data=data_yaml, epochs=epochs, imgsz=imgsz, **kwargs)
        return res
    except Exception as e:
        raise RuntimeError(f"Ultralytics training failed: {e}")
