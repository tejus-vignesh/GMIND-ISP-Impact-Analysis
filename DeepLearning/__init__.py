"""DeepLearning package: Multi-backend object detection training framework.

Provides unified interface for training and benchmarking object detection models
from multiple backends: TorchVision, Ultralytics, and MMDetection.

Main entry points:
    - train_models: Core training script with CLI interface
    - adapters: Backend-specific model loading adapters

Example usage:
    from DeepLearning.train_models import get_model

    # Load a model (auto-detects backend)
    model = get_model('fasterrcnn_resnet50_fpn', num_classes=80)

    # Or train via CLI
    # python -m DeepLearning.train_models --data /path/to/coco --epochs 12
"""

__version__ = "1.0.0"
__author__ = "GMIND Team"

# Core modules
from . import adapters

__all__ = [
    "adapters",
]
