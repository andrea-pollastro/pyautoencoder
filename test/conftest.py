import pytest
import torch


@pytest.fixture(autouse=True)
def _grad_mode_isolation():
    """Start every test with gradients enabled and restore the global mode afterwards."""
    previous = torch.is_grad_enabled()
    torch.set_grad_enabled(True)
    yield
    torch.set_grad_enabled(previous)
