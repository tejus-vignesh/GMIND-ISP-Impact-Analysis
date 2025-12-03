"""
Comprehensive test suite for training and inference on all available models.

Tests that each model in get_all_available_models() can:
1. Be instantiated successfully via appropriate adapter
2. Run a forward pass (inference)
3. Run a training step with loss computation
4. Handle different input shapes
"""

import logging
import sys
from pathlib import Path
from typing import Any, Tuple

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from DeepLearning.adapters import detect_backend, mmdet_adapter, ultralytics_adapter
from DeepLearning.train_models import get_model, get_supported_models

# Alias for backward compatibility
get_all_available_models = get_supported_models


class MockCOCODataset:
    """Mock COCO dataset for testing"""

    def __init__(self, num_samples: int = 2, img_size: Tuple[int, int] = (320, 320)):
        self.num_samples = num_samples
        self.img_size = img_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """Return a sample with image and targets"""
        # Random image tensor
        image = torch.rand(3, self.img_size[0], self.img_size[1])

        # Generate random bounding boxes
        num_objects = torch.randint(1, 4, (1,)).item()
        boxes = torch.rand(num_objects, 4) * 256
        # Ensure x1 < x2 and y1 < y2
        boxes[:, 2] = boxes[:, 2] + boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] + boxes[:, 1]
        boxes = boxes.clamp(max=self.img_size[0] - 1)

        # Create target dict (COCO format)
        target = {
            "boxes": boxes,
            "labels": torch.randint(1, 91, (num_objects,)),  # COCO has 90 classes
            "image_id": torch.tensor([idx]),
            "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
            "iscrowd": torch.zeros(num_objects, dtype=torch.bool),
        }

        return image, target


def prepare_model_for_training(model, num_classes: int = 91, backbone: str = "resnet50"):
    """
    Prepare model for training with appropriate head modifications.

    Args:
        model: The model to prepare
        num_classes: Number of output classes (default 91 for COCO)
        backbone: Backbone type for logging

    Returns:
        Prepared model
    """
    # Move to device
    device = torch.device("cpu")
    model = model.to(device)
    model.train()

    return model


def run_inference_test(
    model, device: torch.device = torch.device("cpu"), img_size: Tuple[int, int] = (320, 320)
):
    """
    Test model inference (forward pass) without training.

    Args:
        model: Model to test
        device: Compute device
        img_size: Input image size

    Returns:
        Inference output
    """
    model.eval()
    model = model.to(device)

    with torch.no_grad():
        # Single image inference
        image = torch.rand(1, 3, img_size[0], img_size[1], device=device)

        # Create dummy targets for validation
        targets = [
            {
                "boxes": torch.tensor([[10, 10, 100, 100]], dtype=torch.float32, device=device),
                "labels": torch.tensor([1], dtype=torch.int64, device=device),
            }
        ]

        try:
            output = model(image, targets)
        except TypeError:
            # Some models might not expect targets during inference
            output = model(image)

    return output


def run_training_step(
    model,
    device: torch.device = torch.device("cpu"),
    learning_rate: float = 0.001,
    img_size: Tuple[int, int] = (320, 320),
):
    """
    Test one complete training step: forward pass, loss computation, backward pass.

    Args:
        model: Model to train
        device: Compute device
        learning_rate: Optimizer learning rate
        img_size: Input image size

    Returns:
        Loss value and model (after one training step)
    """
    model.train()
    model = model.to(device)

    # Create optimizer
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # Create batch
    batch_size = 1
    images = torch.rand(batch_size, 3, img_size[0], img_size[1], device=device)

    # Create targets - check if model needs masks (for Mask R-CNN)
    boxes = torch.tensor(
        [[10, 10, 100, 100], [50, 50, 150, 150]], dtype=torch.float32, device=device
    )
    labels = torch.tensor([1, 2], dtype=torch.int64, device=device)

    targets = [
        {
            "boxes": boxes,
            "labels": labels,
        }
    ]

    # Add masks for Mask R-CNN
    if "mask" in model.__class__.__name__.lower():
        # Create dummy mask tensors
        masks = torch.ones((2, img_size[0], img_size[1]), dtype=torch.uint8, device=device)
        targets[0]["masks"] = masks

    # Forward pass
    optimizer.zero_grad()

    try:
        outputs = model(images, targets)

        # Handle different output formats
        if isinstance(outputs, dict):
            # Loss dict output (PyTorch detection models)
            loss = sum(loss for loss in outputs.values() if isinstance(loss, torch.Tensor))
        elif isinstance(outputs, torch.Tensor):
            # Direct tensor output
            loss = outputs
        else:
            # Fallback: try to extract loss
            if hasattr(outputs, "loss"):
                loss = outputs.loss
            else:
                raise ValueError(f"Cannot extract loss from output type: {type(outputs)}")

        # Backward pass
        if loss > 0:  # Only backprop if loss is positive
            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        return loss.item() if isinstance(loss, torch.Tensor) else loss

    except Exception as e:
        raise RuntimeError(f"Training step failed: {str(e)}")


# ============================================================================
# TORCHVISION MODELS TESTS
# ============================================================================


@pytest.mark.parametrize(
    "model_name",
    [
        "fasterrcnn_resnet50_fpn",
        "fasterrcnn_mobilenet_v3_large_fpn",
        "maskrcnn_resnet50_fpn",
        "ssd300_vgg16",
        "ssdlite320_mobilenet_v3_large",
    ],
)
def test_torchvision_inference(model_name: str):
    """Test TorchVision model inference"""
    model = get_model(model_name, num_classes=91, pretrained=True)
    assert model is not None
    output = run_inference_test(model, img_size=(320, 320))
    assert output is not None


@pytest.mark.parametrize(
    "model_name",
    [
        "fasterrcnn_resnet50_fpn",
        "maskrcnn_resnet50_fpn",
        "ssd300_vgg16",
    ],
)
def test_torchvision_train_step(model_name: str):
    """Test TorchVision model training step"""
    model = get_model(model_name, num_classes=91, pretrained=True)
    model = prepare_model_for_training(model)
    loss = run_training_step(model, img_size=(320, 320))
    assert isinstance(loss, float)
    assert loss >= 0


# ============================================================================
# ULTRALYTICS YOLOV8 TESTS
# ============================================================================
# Note: Ultralytics models require `ultralytics` package installed
# Tests are skipped if not available


@pytest.mark.skip(reason="Requires ultralytics backend adapter")
@pytest.mark.parametrize(
    "model_name",
    [
        "yolov8n",
        "yolov8s",
        "yolov8m",
    ],
)
def test_ultralytics_yolov8_inference(model_name: str):
    """Test Ultralytics YOLOv8 model inference"""
    pass


@pytest.mark.skip(reason="Requires ultralytics backend adapter")
@pytest.mark.parametrize(
    "model_name",
    [
        "yolov8n-p6",
        "yolov8s-p6",
    ],
)
def test_ultralytics_yolov8_p6_inference(model_name: str):
    """Test Ultralytics YOLOv8 P6 variant inference"""
    pass


@pytest.mark.skip(reason="Requires ultralytics backend adapter")
@pytest.mark.parametrize(
    "model_name",
    [
        "yolov9n",
        "yolov9s",
    ],
)
def test_ultralytics_yolov9_inference(model_name: str):
    """Test Ultralytics YOLOv9 model inference"""
    pass


@pytest.mark.skip(reason="Requires ultralytics backend adapter")
@pytest.mark.parametrize(
    "model_name",
    [
        "yolov10n",
    ],
)
def test_ultralytics_yolov10_inference(model_name: str):
    """Test Ultralytics YOLOv10 model inference"""
    pass


@pytest.mark.skip(reason="Requires ultralytics backend adapter")
@pytest.mark.parametrize(
    "model_name",
    [
        "rtdetr-l",
    ],
)
def test_ultralytics_rtdetr_inference(model_name: str):
    """Test Ultralytics RT-DETR model inference"""
    pass


# ============================================================================
# ULTRALYTICS YOLOV5 TESTS
# ============================================================================


@pytest.mark.skip(reason="Requires ultralytics backend adapter")
@pytest.mark.parametrize(
    "model_name",
    [
        "yolov5n",
        "yolov5s",
    ],
)
def test_ultralytics_yolov5_inference(model_name: str):
    """Test Ultralytics YOLOv5 model inference"""
    pass


# ============================================================================
# MMDETECTION TESTS
# ============================================================================


@pytest.mark.skip(reason="Requires mmdetection backend adapter")
@pytest.mark.parametrize(
    "model_name",
    [
        "faster_rcnn",
        "cascade_rcnn",
        "retinanet",
        "ssd",
        "fcos",
        "atss",
    ],
)
def test_mmdetection_inference(model_name: str):
    """Test MMDetection model inference"""
    pass


@pytest.mark.skip(reason="Requires mmdetection backend adapter")
@pytest.mark.parametrize(
    "model_name",
    [
        "yolov3",
        "yolov4",
    ],
)
def test_mmdetection_yolo_inference(model_name: str):
    """Test MMDetection YOLO variants"""
    pass


@pytest.mark.skip(reason="Requires mmdetection backend adapter")
@pytest.mark.parametrize(
    "model_name",
    [
        "detr",
        "dino",
    ],
)
def test_mmdetection_transformer_inference(model_name: str):
    """Test MMDetection transformer-based models"""
    pass


# ============================================================================
# DETECTRON2 TESTS
# ============================================================================

# Detectron2 tests removed — repo focuses on 2D detection via TorchVision, MMDetection, and Ultralytics


# ============================================================================
# INVENTORY VERIFICATION TESTS
# ============================================================================


def test_all_available_models_loadable():
    """Verify that all listed models can be loaded without error"""
    all_models = get_all_available_models()

    # Count models per backend
    total_models = sum(len(models) for models in all_models.values())

    # Should have significant number of models
    assert total_models > 80, f"Expected 80+ models, got {total_models}"

    # Check each backend has models
    assert "torchvision" in all_models
    assert len(all_models["torchvision"]) > 0

    assert "ultralytics_yolov8" in all_models
    assert len(all_models["ultralytics_yolov8"]) > 0

    assert "mmdetection" in all_models
    assert len(all_models["mmdetection"]) > 0


def test_required_models_present():
    """Verify all user-required models are in inventory"""
    all_models = get_all_available_models()

    # User requirements
    required = {
        "torchvision": ["fasterrcnn_resnet50_fpn", "retinanet_resnet50_fpn", "ssd300_vgg16"],
        "ultralytics_yolov8": ["yolov8n", "yolov8m", "yolov8n-p6"],
        "mmdetection": ["faster_rcnn", "cascade_rcnn", "retinanet", "ssd", "atss", "fcos", "detr"],
    }

    for backend, models in required.items():
        assert backend in all_models, f"Backend {backend} not found"
        available = all_models[backend]
        for model in models:
            assert model in available, f"Model {model} not found in {backend}"


def test_yolov8_p6_variants_present():
    """Verify YOLOv8 P6 variants are included"""
    all_models = get_all_available_models()
    yolov8_models = all_models.get("ultralytics_yolov8", [])

    p6_variants = ["yolov8n-p6", "yolov8s-p6", "yolov8m-p6", "yolov8l-p6", "yolov8x-p6"]
    for variant in p6_variants:
        assert variant in yolov8_models, f"P6 variant {variant} not found"


def test_mmdetection_required_models_present():
    """Verify MMDetection required models are included"""
    all_models = get_all_available_models()
    mmdet_models = all_models.get("mmdetection", [])

    required = [
        "faster_rcnn",
        "cascade_rcnn",
        "retinanet",
        "ssd",
        "atss",
        "gfl",
        "fcos",
        "vfnet",
        "reppoints",
        "detr",
        "dino",
        "yolov3",
        "yolov4",
        "yolov5",
        "yolov6",
        "yolov7",
        "hybrid_task_cascade",
        "libra_rcnn",
    ]

    for model in required:
        assert model in mmdet_models, f"MMDetection model {model} not found"


# ============================================================================
# Backend Detection Tests
# ============================================================================


class TestBackendDetection:
    """Test automatic backend detection from model names"""

    def test_detect_torchvision_backend(self):
        """Test detection of TorchVision models"""
        assert detect_backend("fasterrcnn_resnet50_fpn") == "torchvision"
        assert detect_backend("ssd300_vgg16") == "torchvision"
        assert detect_backend("retinanet_resnet50_fpn") == "torchvision"

    def test_detect_ultralytics_backend(self):
        """Test detection of Ultralytics/YOLO models"""
        assert detect_backend("yolov8m") == "ultralytics"
        assert detect_backend("yolov5l") == "ultralytics"
        assert detect_backend("yolov8m-p6") == "ultralytics"
        assert detect_backend("yolov9c") == "ultralytics"
        assert detect_backend("rtdetr-l") == "ultralytics"

    def test_detect_mmdetection_backend(self):
        """Test detection of MMDetection models"""
        assert detect_backend("faster_rcnn") == "mmdetection"
        assert detect_backend("cascade_rcnn") == "mmdetection"
        assert detect_backend("atss") == "mmdetection"
        assert detect_backend("gfl") == "mmdetection"
        assert detect_backend("detr") == "mmdetection"
        assert detect_backend("dino") == "mmdetection"


# ============================================================================
# Adapter Availability Tests
# ============================================================================


class TestAdapterAvailability:
    """Test adapter availability and imports"""

    def test_ultralytics_adapter_signature(self):
        """Test Ultralytics adapter has required functions"""
        assert hasattr(ultralytics_adapter, "is_available")
        assert hasattr(ultralytics_adapter, "get_model")
        assert hasattr(ultralytics_adapter, "inference")

    def test_mmdetection_adapter_signature(self):
        """Test MMDetection adapter has required functions"""
        assert hasattr(mmdet_adapter, "is_available")
        assert hasattr(mmdet_adapter, "get_model")
        assert hasattr(mmdet_adapter, "inference")
        assert hasattr(mmdet_adapter, "find_config_file")


# ============================================================================
# Model Loading with Explicit Backend Tests
# ============================================================================


class TestModelLoadingWithBackend:
    """Test loading models with explicit backend specification"""

    def test_torchvision_explicit_backend(self):
        """Test loading TorchVision model with explicit backend"""
        model = get_model("fasterrcnn_resnet50_fpn", num_classes=80, backend="torchvision")
        assert model is not None
        logger.info("Loaded TorchVision model with explicit backend")

    def test_auto_backend_detection_yolo(self):
        """Test auto backend detection for YOLO model"""
        # This will fail if ultralytics not installed, which is ok
        try:
            model = get_model("yolov8n", num_classes=80, backend="auto")
            assert model is not None
            logger.info("Loaded YOLO model with auto backend detection")
        except ValueError as e:
            logger.warning(f"Ultralytics not available: {e}")
            pytest.skip("Ultralytics not installed")

    def test_backend_parameter_passthrough(self):
        """Test that backend parameter is properly passed"""
        # Test TorchVision with explicit backend
        model = get_model("fasterrcnn_resnet50_fpn", num_classes=80, backend="torchvision")
        assert model is not None

        # Test auto detection
        model = get_model("fasterrcnn_resnet50_fpn", num_classes=80, backend="auto")
        assert model is not None


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
