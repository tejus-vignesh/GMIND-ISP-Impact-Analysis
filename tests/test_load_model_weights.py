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
import os
import tempfile

import torch

from DeepLearning.train_models import load_model_weights


def test_load_model_weights_variants():
    # create a simple model and save different checkpoint formats
    model = torch.nn.Linear(2, 2)
    state = model.state_dict()

    with tempfile.TemporaryDirectory() as d:
        # raw state_dict
        p1 = os.path.join(d, "state.pth")
        torch.save(state, p1)

        # wrapped as model_state
        p2 = os.path.join(d, "wrapped.pth")
        torch.save({"model_state": state}, p2)

        # wrapped as state_dict
        p3 = os.path.join(d, "wrapped2.pth")
        torch.save({"state_dict": state}, p3)

        m1 = torch.nn.Linear(2, 2)
        load_model_weights(m1, p1, torch.device("cpu"))
        # loaded values should match
        for k, v in m1.state_dict().items():
            assert torch.allclose(v, state[k])

        m2 = torch.nn.Linear(2, 2)
        load_model_weights(m2, p2, torch.device("cpu"))
        for k, v in m2.state_dict().items():
            assert torch.allclose(v, state[k])

        m3 = torch.nn.Linear(2, 2)
        load_model_weights(m3, p3, torch.device("cpu"))
        for k, v in m3.state_dict().items():
            assert torch.allclose(v, state[k])
