import logging
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# If torch/torchvision are not installed in the test environment, add light-weight
# stubs so importing `DeepLearning.train_models` succeeds for tests that don't
# require full torch functionality.
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

from DeepLearning.train_models import get_supported_models


def test_supported_models_nonempty():
    models = get_supported_models()
    logger.info("Supported models (%d total):", len(models))
    for model in models:
        logger.info("  - %s", model)
    assert isinstance(models, list)
    assert "fasterrcnn_resnet50_fpn" in models
    assert len(models) >= 1
