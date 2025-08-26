from typing import Any
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass(slots=True)
class ModelOutput(ABC):
    """Marker base class for all model outputs."""
    pass

class BaseAutoencoder(nn.Module, ABC):
    """
    Base class for Autoencoders (deterministic, variational, flow-based, etc.).

    Training (grad-enabled; subclasses decide the exact ModelOutput fields):
      - _encode(x, **kwargs) -> ModelOutput
      - _decode(z, **kwargs) -> ModelOutput
      - forward(x, **kwargs) -> ModelOutput

    Inference (no grad; explicit decode(z) contract):
      - encode(x, use_eval=True, **kwargs) -> ModelOutput
      - decode(z, use_eval=True, **kwargs) -> ModelOutput
    """

    # --- gradient-enabled training APIs (to be implemented by subclasses) ---
    @abstractmethod
    def _encode(self, x: torch.Tensor, **kwargs: Any) -> ModelOutput:
        """Returns an encode-time ModelOutput (e.g., must include at least .z)."""
        pass

    @abstractmethod
    def _decode(self, z: torch.Tensor, **kwargs: Any) -> ModelOutput:
        """Returns a decode-time ModelOutput (e.g., must include at least .x_hat)."""
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs: Any) -> ModelOutput:
        """Returns a forward-time ModelOutput (typically includes .x_hat and .z)."""
        pass

    # --- inference-only convenience wrappers (no grad; optional eval mode) ---
    @torch.inference_mode()
    def encode(self, x: torch.Tensor, use_eval: bool = True, **kwargs: Any) -> ModelOutput:
        if not use_eval:
            return self._encode(x, **kwargs)
        prev = self.training
        try:
            self.eval()
            return self._encode(x, **kwargs)
        finally:
            self.train(prev)

    @torch.inference_mode()
    def decode(self, z: torch.Tensor, use_eval: bool = True, **kwargs: Any) -> ModelOutput:
        if not use_eval:
            return self._decode(z, **kwargs)
        prev = self.training
        try:
            self.eval()
            return self._decode(z, **kwargs)
        finally:
            self.train(prev)
