from typing import Any, Mapping
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from functools import wraps

class NotBuiltError(RuntimeError): pass

def _make_guard(name: str, orig):
    @wraps(orig)
    def guarded(self, *args, **kwargs):
        if not getattr(self, "_built", False):
            raise RuntimeError("Model is not built. Call `build(x)` first.")
        # first post-build call: swap in the original method on THIS instance
        bound_orig = orig.__get__(self, self.__class__)
        setattr(self, name, bound_orig)
        return bound_orig(*args, **kwargs)
    return guarded

@dataclass(slots=True)
class ModelOutput(ABC):
    """Marker base class for all model outputs with a smart repr."""

    def __repr__(self) -> str:
        parts = []
        for f in fields(self):
            name = f.name
            value = getattr(self, name)
            if isinstance(value, torch.Tensor):
                parts.append(
                    f"{name}=Tensor(shape={tuple(value.shape)}, dtype={value.dtype})"
                )
            else:
                s = repr(value)
                parts.append(f"{name}={s}")
        return f"{self.__class__.__name__}({', '.join(parts)})"

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

    _GUARDED = {"forward", "_encode", "_decode"}
    
    def __init__(self) -> None:
        super().__init__()
        self._built: bool = False

    def __init_subclass__(cls, **kwargs):
        """Add guards and auto-warmup after build()."""
        super().__init_subclass__(**kwargs)
        
        # 1) Guard forward/_encode/_decode until built; self-remove on first call after build.
        for name in cls._GUARDED:
            if name in cls.__dict__:
                orig = cls.__dict__[name]
                setattr(cls, name, _make_guard(name, orig))

        # 2) Wrap subclass build(x) to enforce _built and run a tiny forward warm-up.
        if "build" in cls.__dict__:
            _orig_build = cls.__dict__["build"]

            @wraps(_orig_build)
            def _wrapped_build(self, input_sample: torch.Tensor) -> None:
                # Ensure no grad & call user build
                with torch.no_grad():
                    _orig_build(self, input_sample)
                if not getattr(self, "_built", False):
                    raise RuntimeError("Subclass build(x) must set `self._built = True` once building is done.")

                # Try a cheap warm-up forward to drop the guards (and catch obvious wiring issues).
                # Use a 1-sample slice when possible.
                try:
                    _ = self.forward(input_sample)  # first call will swap out guards on this instance
                except TypeError:
                    # If forward requires extra non-default args, just skip the warm-up.
                    pass

            setattr(cls, "build", _wrapped_build)

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
        """Returns an encode-time ModelOutput without gradient and eventually with use_eval."""
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
        """Returns an decode-time ModelOutput without gradient and eventually with use_eval."""
        if not use_eval:
            return self._decode(z, **kwargs)
        prev = self.training
        try:
            self.eval()
            return self._decode(z, **kwargs)
        finally:
            self.train(prev)
    
    # ----- required build step -----
    @abstractmethod
    def build(self, input_sample: torch.Tensor) -> None:
        """
        Prepare the module (e.g., create size-dependent layers, buffers, dtype/device align).
        Must set `self._built = True` when ready.
        """
        pass

    @property
    def built(self) -> bool:
        return self._built
    
    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        if not self._built:
            raise NotBuiltError(
                "load_state_dict called before build(). "
                "Call model.build(example_x) so parameters exist, then load."
            )
        return super().load_state_dict(state_dict=state_dict, strict=strict, assign=assign)
        