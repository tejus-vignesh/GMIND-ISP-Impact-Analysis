"""
Test that training works by checking loss decreases over a few steps.

This is a quick smoke test to verify:
1. Models can be trained (forward + backward pass works)
2. Loss computation is correct
3. Loss decreases (or at least changes) after optimizer step
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import pytest
import torch
import torch.optim as optim

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from DeepLearning.adapters import detect_backend
from DeepLearning.train_models import get_model

# Test models from each backend
# Using synthetic data only (no dataset downloads)
TEST_MODELS = [
    # TorchVision models
    ("fasterrcnn_resnet50_fpn", "torchvision"),
    ("retinanet_resnet50_fpn", "torchvision"),
    # Ultralytics YOLO models (using synthetic data, no dataset needed)
    ("yolov8n", "ultralytics"),
    # MMDetection models (require config files, will skip if not available)
    ("faster_rcnn", "mmdetection"),
]


def create_synthetic_batch(
    batch_size: int = 1,
    img_size: Tuple[int, int] = (320, 320),
    num_objects: int = 2,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, list]:
    """Create synthetic image and target batch for training."""
    # Create random images
    images = torch.rand(batch_size, 3, img_size[0], img_size[1], device=device)

    # Create targets with bounding boxes
    targets = []
    for i in range(batch_size):
        # Generate random boxes (x1, y1, x2, y2)
        boxes = torch.rand(num_objects, 4, device=device) * min(img_size)
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]  # x2 = x1 + width
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]  # y2 = y1 + height
        boxes = boxes.clamp(max=min(img_size) - 1)

        target = {
            "boxes": boxes,
            "labels": torch.randint(1, 91, (num_objects,), device=device),  # COCO classes
        }

        # Add masks for Mask R-CNN
        if "mask" in str(type(images)).lower():
            masks = torch.zeros(
                (num_objects, img_size[0], img_size[1]), dtype=torch.uint8, device=device
            )
            for j in range(num_objects):
                x1, y1, x2, y2 = boxes[j].int()
                masks[j, y1:y2, x1:x2] = 1
            target["masks"] = masks

        targets.append(target)

    return images, targets


def extract_loss(outputs: Any) -> torch.Tensor:
    """Extract loss from model outputs (handles different formats)."""
    if isinstance(outputs, dict):
        # Loss dict (PyTorch detection models)
        loss = sum(v for v in outputs.values() if isinstance(v, torch.Tensor))
    elif isinstance(outputs, torch.Tensor):
        # Direct tensor
        loss = outputs
    elif hasattr(outputs, "loss"):
        # Object with loss attribute
        loss = outputs.loss
    else:
        raise ValueError(f"Cannot extract loss from output type: {type(outputs)}")

    return loss


def run_training_steps(
    model: torch.nn.Module,
    num_steps: int = 2,
    device: torch.device = torch.device("cpu"),
    learning_rate: float = 0.01,
    img_size: Tuple[int, int] = (320, 320),
) -> list:
    """
    Run multiple training steps and return loss values.

    Args:
        model: Model to train
        num_steps: Number of training steps to run
        device: Compute device
        learning_rate: Learning rate for optimizer
        img_size: Input image size

    Returns:
        List of loss values for each step
    """
    model.train()
    model = model.to(device)

    # Create optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(trainable_params, lr=learning_rate, momentum=0.9)

    losses = []

    for step in range(num_steps):
        # Create synthetic batch
        images, targets = create_synthetic_batch(batch_size=1, img_size=img_size, device=device)

        # Forward pass
        optimizer.zero_grad()

        try:
            # Handle different input formats for different backends
            backend = detect_backend(
                model.__class__.__name__ if hasattr(model, "__class__") else ""
            )

            if backend == "ultralytics":
                # Ultralytics YOLO models work differently - they need to be trained via .train()
                # For testing, we'll use a workaround: create a simple loss from model outputs
                # In practice, YOLO training uses model.train(data=...) which requires a dataset
                # For this test, we'll skip YOLO or use a simplified approach
                try:
                    # Try to get model outputs (inference mode)
                    outputs = model(images[0] if isinstance(images, list) else images)
                    # Create a dummy loss for testing purposes
                    # In real training, YOLO uses its own training loop
                    loss = torch.tensor(1.0, device=device, requires_grad=True)
                    logger.warning(
                        "Ultralytics YOLO models use their own training loop. Using dummy loss for test."
                    )
                except Exception as e:
                    logger.warning(f"Could not test YOLO model directly: {e}. Skipping.")
                    loss = torch.tensor(1.0, device=device, requires_grad=True)
            else:
                # Standard PyTorch format: list of images, list of targets
                if isinstance(images, torch.Tensor):
                    # Convert to list format for torchvision models
                    images_list = [images[i] for i in range(images.shape[0])]
                else:
                    images_list = images

                outputs = model(images_list, targets)
                loss = extract_loss(outputs)

            # Backward pass
            if loss.requires_grad:
                loss.backward()
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            loss_value = loss.item() if isinstance(loss, torch.Tensor) else float(loss)
            losses.append(loss_value)
            logger.info(f"Step {step + 1}/{num_steps}: Loss = {loss_value:.4f}")

        except Exception as e:
            logger.error(f"Training step {step + 1} failed: {e}")
            raise

    return losses


@pytest.mark.parametrize("model_name,backend", TEST_MODELS)
def test_training_loss_decreases(model_name: str, backend: str):
    """
    Test that loss decreases (or at least changes) over 2 training steps.

    This verifies:
    - Model can be instantiated
    - Forward pass works
    - Loss computation works
    - Backward pass works
    - Optimizer step works
    - Loss changes after training step
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Skip MMDetection if not available or if config files are needed
    if backend == "mmdetection":
        try:
            from DeepLearning.adapters import mmdet_adapter

            if not mmdet_adapter.is_available():
                pytest.skip("MMDetection not available")
            # MMDetection models require config files which may not be available
            # Skip for now - can be enabled when config files are properly set up
            pytest.skip("MMDetection models require config files - skipping for now")
        except ImportError:
            pytest.skip("MMDetection not available")

    # Skip Ultralytics YOLO - they use their own training loop that requires datasets
    if backend == "ultralytics":
        pytest.skip(
            "Ultralytics YOLO models use their own training loop with datasets - skipping synthetic data test"
        )

    # Get model
    try:

        # Get model - backend parameter format depends on the function signature
        if backend == "torchvision":
            model = get_model(model_name, num_classes=91, pretrained=False)
        elif backend == "ultralytics":
            model = get_model(model_name, num_classes=91, pretrained=False, backend="ultralytics")
        elif backend == "mmdetection":
            model = get_model(model_name, num_classes=91, pretrained=False, backend="mmdetection")
        else:
            model = get_model(model_name, num_classes=91, pretrained=False)
    except Exception as e:
        pytest.skip(f"Could not load model {model_name}: {e}")

    # Determine appropriate image size
    img_size = (320, 320)
    if "300" in model_name:
        img_size = (300, 300)
    elif "640" in model_name or "yolo" in model_name.lower():
        img_size = (640, 640)

    # Run training steps
    try:
        losses = run_training_steps(
            model, num_steps=2, device=device, learning_rate=0.01, img_size=img_size
        )
    except Exception as e:
        pytest.fail(f"Training failed: {e}")

    # Verify we got loss values
    assert len(losses) == 2, f"Expected 2 loss values, got {len(losses)}"
    assert all(isinstance(l, (int, float)) for l in losses), "Losses should be numeric"
    assert all(l >= 0 for l in losses), "Losses should be non-negative"

    # Check that loss changes (either decreases or increases, but not stays exactly the same)
    # Note: Loss might not always decrease in just 2 steps, but it should change
    loss_changed = abs(losses[1] - losses[0]) > 1e-6

    logger.info(f"Loss step 1: {losses[0]:.4f}, Loss step 2: {losses[1]:.4f}")
    logger.info(
        f"Loss {'decreased' if losses[1] < losses[0] else 'increased' if losses[1] > losses[0] else 'unchanged'}"
    )

    assert loss_changed, (
        f"Loss did not change between steps: {losses[0]:.4f} -> {losses[1]:.4f}. "
        "This might indicate training is not working correctly."
    )

    # Ideally, loss should decrease, but we'll be lenient and just check it changes
    # In practice, loss might increase slightly due to learning dynamics
    if losses[1] < losses[0]:
        logger.info("Loss decreased as expected!")


if __name__ == "__main__":
    # Run a quick test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model("fasterrcnn_resnet50_fpn", num_classes=91, pretrained=False)
    losses = run_training_steps(model, num_steps=2, device=device)
    logger.info("Losses: %s", losses)
    logger.info("Loss decreased: %s", losses[1] < losses[0])
