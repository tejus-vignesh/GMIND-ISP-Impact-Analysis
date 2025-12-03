import importlib.util
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
try:
    import torch

    # if torch is a stub inserted into sys.modules it will usually not have a __file__ attribute
    if getattr(torch, "__file__", None) is None:
        raise ImportError
    import torchvision

    if getattr(torchvision, "__file__", None) is None:
        raise ImportError
except Exception:
    import pytest

    pytest.skip("real torch/torchvision not available in environment", allow_module_level=True)
from DeepLearning import train_models


@pytest.mark.parametrize("model_name", train_models.get_supported_models())
def test_model_build_and_forward(model_name):
    # Build model without pretrained weights to avoid downloads
    model = train_models.get_model(model_name, num_classes=5, pretrained=False)
    assert isinstance(model, torch.nn.Module)

    model.eval()
    # choose an input size that's reasonable for the model
    size = 224
    if "ssd300" in model_name:
        size = 300
    if "ssdlite320" in model_name or "320" in model_name:
        size = 320

    imgs = [torch.rand(3, size, size)]

    # Run forward (detection models accept list[Tensor] as input for inference)
    with torch.no_grad():
        outputs = model(imgs)

    # Outputs should be a list or tuple of dicts for torchvision detectors
    assert isinstance(outputs, (list, tuple))
    assert len(outputs) >= 1
