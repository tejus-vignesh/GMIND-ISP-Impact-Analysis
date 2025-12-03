import builtins
import pathlib
import sys
import types

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# Provide light-weight torch/torchvision stubs if not present so tests can import module.
try:
    import torch  # type: ignore
except Exception:
    import types

    fake_torch = types.ModuleType("torch")
    fake_torch.nn = types.ModuleType("torch.nn")
    fake_torch.nn.Module = object
    fake_torch.optim = types.ModuleType("torch.optim")
    fake_torch.optim.Optimizer = object
    fake_torch.utils = types.ModuleType("torch.utils")
    fake_torch.utils.data = types.ModuleType("torch.utils.data")
    fake_torch.utils.data.DataLoader = lambda *a, **k: None
    fake_torch.as_tensor = lambda x, dtype=None: x
    fake_torch.empty = lambda shape: []
    fake_torch.tensor = lambda x: x
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    fake_torch.device = lambda x: types.SimpleNamespace(type="cpu")
    sys.modules["torch"] = fake_torch
    sys.modules["torch.nn"] = fake_torch.nn
    sys.modules["torch.optim"] = fake_torch.optim
    sys.modules["torch.utils"] = fake_torch.utils
    sys.modules["torch.utils.data"] = fake_torch.utils.data

try:
    import torchvision  # type: ignore
except Exception:
    import types

    fake_torchvision = types.ModuleType("torchvision")
    fake_torchvision.models = types.SimpleNamespace(detection=types.SimpleNamespace())
    # also provide submodules that train_models imports
    fake_torchvision_models = types.ModuleType("torchvision.models")
    fake_torchvision_models.detection = types.SimpleNamespace()
    fake_torchvision_datasets = types.ModuleType("torchvision.datasets")

    class CocoDetection:
        def __init__(self, img_folder, ann_file):
            self.ids = []

        def __getitem__(self, idx):
            raise IndexError

    fake_torchvision_datasets.CocoDetection = CocoDetection
    fake_torchvision_ops = types.ModuleType("torchvision.ops")
    fake_torchvision_ops.boxes = types.SimpleNamespace()
    sys.modules["torchvision.models"] = fake_torchvision_models
    sys.modules["torchvision.models.detection"] = fake_torchvision_models.detection
    sys.modules["torchvision.datasets"] = fake_torchvision_datasets
    sys.modules["torchvision.ops"] = fake_torchvision_ops

    # minimal transforms shim used by train_models.get_transform
    class ToTensor:
        def __call__(self, x):
            return x

    class RandomHorizontalFlip:
        def __init__(self, p=0.5):
            self.p = p

        def __call__(self, x):
            return x

    class Compose:
        def __init__(self, ops):
            self.ops = ops

        def __call__(self, x):
            for o in self.ops:
                x = o(x)
            return x

    fake_torchvision.transforms = types.SimpleNamespace(
        ToTensor=ToTensor, RandomHorizontalFlip=RandomHorizontalFlip, Compose=Compose
    )
    sys.modules["torchvision"] = fake_torchvision

import pytest

from DeepLearning import train_models


def test_evaluate_no_pycocotools(monkeypatch, capsys):
    # Force imports of pycocotools to fail by patching __import__ for that name
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("pycocotools"):
            raise ImportError("no module named pycocotools")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    # Evaluate should return None and print a message
    res = train_models.evaluate_coco(model=None, data_loader=[], device="cpu")
    assert res is None
    # restore import done automatically by monkeypatch teardown
