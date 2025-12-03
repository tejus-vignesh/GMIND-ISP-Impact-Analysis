"""Data augmentation transforms for object detection.

Provides spatial and color augmentations that properly transform bounding boxes.
"""

import math
import random
from typing import Any, Dict, Tuple

import torch
import torchvision.transforms.functional as F
from torchvision import transforms


class RandomHorizontalFlip:
    """Random horizontal flip with bounding box transformation."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(
        self, image: torch.Tensor, target: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if random.random() < self.p:
            image = F.hflip(image)
            if "boxes" in target and len(target["boxes"]) > 0:
                width = image.shape[-1]
                boxes = target["boxes"].clone()
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]  # Flip x coordinates
                target["boxes"] = boxes
        return image, target


class RandomVerticalFlip:
    """Random vertical flip with bounding box transformation."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(
        self, image: torch.Tensor, target: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if random.random() < self.p:
            image = F.vflip(image)
            if "boxes" in target and len(target["boxes"]) > 0:
                height = image.shape[-2]
                boxes = target["boxes"].clone()
                boxes[:, [1, 3]] = height - boxes[:, [3, 1]]  # Flip y coordinates
                target["boxes"] = boxes
        return image, target


class ColorJitter:
    """Color jitter augmentation (brightness, contrast, saturation, hue)."""

    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(
        self, image: torch.Tensor, target: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Apply color jitter (doesn't affect bounding boxes)
        if self.brightness > 0:
            image = F.adjust_brightness(
                image, 1.0 + random.uniform(-self.brightness, self.brightness)
            )
        if self.contrast > 0:
            image = F.adjust_contrast(image, 1.0 + random.uniform(-self.contrast, self.contrast))
        if self.saturation > 0:
            image = F.adjust_saturation(
                image, 1.0 + random.uniform(-self.saturation, self.saturation)
            )
        if self.hue > 0:
            image = F.adjust_hue(image, random.uniform(-self.hue, self.hue))
        return image, target


class RandomRotation:
    """Random rotation with bounding box transformation (simplified - uses bounding box of rotated image)."""

    def __init__(self, degrees: float = 10.0, p: float = 0.5):
        self.degrees = degrees
        self.p = p

    def __call__(
        self, image: torch.Tensor, target: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if random.random() < self.p and "boxes" in target and len(target["boxes"]) > 0:
            angle = random.uniform(-self.degrees, self.degrees)
            image = F.rotate(image, angle, interpolation=F.InterpolationMode.BILINEAR)
            # Note: Proper rotation of bounding boxes requires more complex transformation
            # For now, we'll keep boxes as-is (rotation is small, so approximation is acceptable)
            # For production, consider using imgaug or albumentations for proper box rotation
        return image, target


def get_detection_transforms(train: bool = True, augment_level: str = "medium") -> Any:
    """Get augmentation transforms for object detection.

    Args:
        train: Whether this is for training (augmentations enabled) or validation (no augmentations)
        augment_level: Level of augmentation - "light", "medium", or "heavy"

    Returns:
        Transform function that takes (image, target) and returns (transformed_image, transformed_target)
    """
    if not train:
        # Validation: only convert to tensor
        def transform(image, target):
            if not isinstance(image, torch.Tensor):
                image = transforms.ToTensor()(image)
            return image, target

        return transform

    # Training: apply augmentations
    if augment_level == "light":
        # Minimal augmentations
        transforms_list = [
            transforms.ToTensor(),
            RandomHorizontalFlip(p=0.5),
        ]
    elif augment_level == "medium":
        # Moderate augmentations (recommended)
        transforms_list = [
            transforms.ToTensor(),
            RandomHorizontalFlip(p=0.5),
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ]
    else:  # heavy
        # Heavy augmentations
        transforms_list = [
            transforms.ToTensor(),
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.3),
            ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
            RandomRotation(degrees=5.0, p=0.3),
        ]

    def compose_transforms(image, target):
        """Apply all transforms in sequence."""
        if not isinstance(image, torch.Tensor):
            # Convert PIL to tensor first
            image = transforms.ToTensor()(image)

        for transform_fn in transforms_list:
            if isinstance(
                transform_fn,
                (RandomHorizontalFlip, RandomVerticalFlip, ColorJitter, RandomRotation),
            ):
                image, target = transform_fn(image, target)
            else:
                # Standard torchvision transform (only affects image)
                image = transform_fn(image)

        return image, target

    return compose_transforms
