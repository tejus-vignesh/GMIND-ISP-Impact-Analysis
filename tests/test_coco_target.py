import importlib.util
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
try:
    import torch

    if getattr(torch, "__file__", None) is None:
        raise ImportError
except Exception:
    import pytest

    pytest.skip("real torch not available in environment", allow_module_level=True)
import torch

from DeepLearning.train_models import coco_to_target


def test_coco_to_target_single_box():
    ann = [{"bbox": [10, 20, 30, 40], "category_id": 3, "area": 1200, "iscrowd": 0}]
    target = coco_to_target(ann, image_id=5, image_size=(100, 100))

    assert "boxes" in target and "labels" in target and "image_id" in target
    assert target["boxes"].shape == (1, 4)
    assert torch.allclose(target["boxes"][0], torch.tensor([10.0, 20.0, 40.0, 60.0]))
    assert target["labels"].item() == 3
    assert target["area"].item() == 1200.0
    assert int(target["image_id"].item()) == 5
