from typing import Any, Mapping, Callable, Iterable
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from functools import wraps

class NotBuiltError(RuntimeError): 
    pass

def _make_guard(name: str, orig: Callable[..., Any]) -> Callable[..., Any]:
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
    
class BuildGuardMixin(ABC):
    """
    Mixin that:
      - requires subclasses to define a class attribute `_GUARDED`
        (an iterable of method names)
      - wraps those methods with a 'built' guard
      - wraps build(x) to enforce self._built and run a warm-up forward
    """
    def __init__(self):
        super().__init__()
        self._built = False

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)

        # 0) Enforce that subclasses define `_GUARDED`
        guarded = getattr(cls, "_GUARDED", None)
        if guarded is None:
            raise TypeError(
                f"{cls.__name__} must define a class attribute `_GUARDED` "
                "with the names of methods to guard."
            )
        if not isinstance(guarded, Iterable) or isinstance(guarded, (str, bytes)):
            raise TypeError(
                f"{cls.__name__}._GUARDED must be an iterable of method names, "
                f"got {type(guarded).__name__}."
            )

        # Normalize to a set of strings
        guarded = set(guarded)
        setattr(cls, "_GUARDED", guarded)

        # 1) Guard methods in _GUARDED until built; self-remove on first call
        for name in guarded:
            if name in cls.__dict__:
                orig = cls.__dict__[name]
                setattr(cls, name, _make_guard(name, orig))

        # 2) Wrap subclass build(x) to enforce _built and run a tiny warm-up.
        if "build" in cls.__dict__:
            _orig_build = cls.__dict__["build"]

            @wraps(_orig_build)
            def _wrapped_build(self, input_sample: torch.Tensor) -> None:
                # Ensure no grad & call user build
                with torch.no_grad():
                    _orig_build(self, input_sample)
                if not getattr(self, "_built", False):
                    raise RuntimeError(
                        "Subclass build(x) must set `self._built = True` once building is done."
                    )

                # Try a cheap warm-up forward to drop the guards (and catch obvious wiring issues).
                try:
                    with torch.no_grad():
                        _ = self.forward(input_sample)  # first call will swap out guards on this instance
                except TypeError:
                    # If forward requires extra non-default args, just skip the warm-up.
                    pass

            setattr(cls, "build", _wrapped_build)

class BaseAutoencoder(BuildGuardMixin, nn.Module, ABC):
    """
    Base class for Autoencoders.

    Training (grad-enabled; subclasses decide the exact ModelOutput fields):
      - _encode(x, *args, **kwargs) -> ModelOutput
      - _decode(z, *args, **kwargs) -> ModelOutput
      - forward(x, *args, **kwargs) -> ModelOutput

    Inference (no grad; explicit decode(z) contract):
      - encode(x, use_eval=True, *args, **kwargs) -> ModelOutput
      - decode(z, use_eval=True, *args, **kwargs) -> ModelOutput
    """

    _GUARDED = {"forward", "_encode", "_decode"}
    
    def __init__(self) -> None:
        super().__init__()

    # --- gradient-enabled training APIs (to be implemented by subclasses) ---
    @abstractmethod
    def _encode(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> ModelOutput:
        """Returns an encode-time ModelOutput (e.g., must include at least .z)."""
        pass

    @abstractmethod
    def _decode(self, z: torch.Tensor, *args: Any, **kwargs: Any) -> ModelOutput:
        """Returns a decode-time ModelOutput (e.g., must include at least .x_hat)."""
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> ModelOutput:
        """Returns a forward-time ModelOutput (typically includes .x_hat and .z)."""
        pass

    # --- inference-only convenience wrappers (no grad; optional eval mode) ---
    @torch.inference_mode()
    def encode(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> ModelOutput:
        """Returns an encode-time ModelOutput without gradient."""
        return self._encode(x, *args, **kwargs)

    @torch.inference_mode()
    def decode(self, z: torch.Tensor, *args: Any, **kwargs: Any) -> ModelOutput:
        """Returns an decode-time ModelOutput without gradient."""
        return self._decode(z, *args, **kwargs)
    
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
        