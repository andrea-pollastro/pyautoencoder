from typing import Any
from collections.abc import Callable, Iterable, Mapping
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from functools import wraps

from ..loss.base import LossResult

class NotBuiltError(RuntimeError): 
    """Exception raised when a guarded method is called on a model that has not been built.

    This error is raised by :class:`BuildGuardMixin` when a method
    listed in ``_GUARDED`` is invoked before :meth:`build` has successfully
    completed and set ``self._built = True``.
    """

    pass

@dataclass(slots=True)
class ModelOutput(ABC):
    """Base class for autoencoder outputs with a concise, tensor-aware ``repr``.

    Subclasses are dataclasses that group together tensors and auxiliary values
    produced by a model (for example latent codes, reconstructions, losses, etc.).
    The custom :meth:`__repr__` implementation prints tensor fields using only
    their shape and dtype instead of full values, which keeps logs readable even
    for large tensors.

    Notes
    -----
    Any field whose value is a :class:`torch.Tensor` is rendered as::

        Tensor(shape=(...), dtype=...)

    All other fields are rendered using :func:`repr`.
    """

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
    """Mixin that guards selected methods until the module has been built.

    Classes that inherit from this mixin must define a class attribute
    ``_GUARDED`` containing the **names** of methods that should not be
    callable until :meth:`build` has been run successfully. These methods are
    wrapped at class creation time so that they:

    * Raise :class:`NotBuiltError` when called while ``self._built`` is ``False``.
    * On the first successful call after the model is built, replace the guarded wrapper on that instance with the original method for zero overhead in subsequent calls.

    If the subclass defines a :meth:`build` method, it is also wrapped so that
    it is executed under ``torch.no_grad()`` and is required to set
    ``self._built = True`` when the module is ready.

    Attributes
    ----------
    _built : bool
        Flag indicating whether :meth:`build` has completed successfully.
    _GUARDED : set[str]
        Names of methods that are wrapped with the build guard. Must be defined
        by subclasses as an iterable of method names.
    """

    def __init__(self):
        super().__init__()
        self._built = False

    @property
    def built(self) -> bool:
        """Whether the module has been successfully built.

        Returns
        -------
        bool
            ``True`` if :meth:`build` has completed and set ``self._built = True``,
            ``False`` otherwise.
        """

        return self._built

    @staticmethod
    def _make_guard(orig: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap ``orig`` so it raises :class:`NotBuiltError` until the model is built."""

        @wraps(orig)
        def guarded(self, *args, **kwargs):
            if not getattr(self, "_built", False):
                raise NotBuiltError("Model is not built. Call `build(x)` first.")
            return orig(self, *args, **kwargs)
        return guarded

    def __init_subclass__(cls, **kwargs) -> None:
        """Hook that installs build guards on subclass methods at class-creation time.

        Subclasses only need to declare **new** method names in ``_GUARDED``; guards
        from all ancestors are accumulated automatically. ``_GUARDED`` may be omitted
        entirely when a subclass adds no new guarded methods.

        ``_GUARDED`` is required on classes that have no ancestor defining it
        (i.e. the root of a guarded hierarchy). Use ``_GUARDED = set()`` if the
        root exposes no guarded methods.
        """

        super().__init_subclass__(**kwargs)

        own_guarded = cls.__dict__.get("_GUARDED", None)

        # Walk the MRO to find the nearest ancestor whose _GUARDED was already
        # accumulated and stamped onto the class by this hook.
        parent_guarded: set[str] = set()
        for base in cls.__mro__[1:]:
            if "_GUARDED" in base.__dict__:
                parent_guarded = set(base.__dict__["_GUARDED"])
                break

        # Require explicit _GUARDED only when no ancestor provides one.
        if own_guarded is None and not parent_guarded:
            raise TypeError(
                f"{cls.__name__} must define a class attribute `_GUARDED` "
                "with the names of methods to guard (use `_GUARDED = set()` if none)."
            )

        if own_guarded is not None:
            if not isinstance(own_guarded, Iterable) or isinstance(own_guarded, (str, bytes)):
                raise TypeError(
                    f"{cls.__name__}._GUARDED must be an iterable of method names, "
                    f"got {type(own_guarded).__name__}."
                )

        full_guarded = set(own_guarded or set()) | parent_guarded
        setattr(cls, "_GUARDED", full_guarded)

        for name in full_guarded:
            if name in cls.__dict__:
                orig = cls.__dict__[name]
                setattr(cls, name, BuildGuardMixin._make_guard(orig))

        if "build" in cls.__dict__:
            _orig_build = cls.__dict__["build"]

            @wraps(_orig_build)
            def _wrapped_build(self, *args: Any, **kwargs: Any) -> None:

                if getattr(self, "_built", False):
                    return

                with torch.no_grad():
                    _orig_build(self, *args, **kwargs)

                if not getattr(self, "_built", False):
                    raise NotBuiltError(
                        "Subclass build(x) must set `self._built = True` once building is done."
                    )

            setattr(cls, "build", _wrapped_build)

class BaseAutoencoder(BuildGuardMixin, nn.Module, ABC):
    """Base class for autoencoders with an explicit build step.

    This class defines a common interface for autoencoders that split their
    logic into *encoding*, *decoding* and *forward* passes, and enforces a
    build step via :class:`BuildGuardMixin`.

    Training API (gradients enabled)
    --------------------------------
    Subclasses must implement the following abstract methods:

    * ``_encode(x, *args, **kwargs) -> ModelOutput``  
      Low-level encoder that typically returns a :class:`ModelOutput` with at
      least a latent code attribute (for example ``z``).

    * ``_decode(z, *args, **kwargs) -> ModelOutput``  
      Low-level decoder that typically returns a :class:`ModelOutput` with at
      least a reconstruction attribute (for example ``x_hat``).

    * ``forward(x, *args, **kwargs) -> ModelOutput``  
      Full forward pass used during training. This usually combines encoding
      and decoding and returns a :class:`ModelOutput` that may contain both
      ``z`` and ``x_hat``, plus any other quantities needed for loss
      computation.

    Inference API (no gradients)
    ----------------------------
    For convenience, the class also exposes high-level inference helpers that
    are executed under :func:`torch.inference_mode`:

    * :meth:`encode` – calls :meth:`_encode` without tracking gradients.
    * :meth:`decode` – calls :meth:`_decode` without tracking gradients.

    Build Contract
    --------------
    Before training or inference, subclasses must implement and call
    :meth:`build` with a representative input tensor. The build method is
    responsible for creating any size-dependent layers or buffers and must set
    ``self._built = True`` when the module is ready. Methods listed in
    ``_GUARDED`` (by default ``{"forward", "_encode", "_decode"}``) are
    protected and will raise :class:`NotBuiltError` if called before the build
    step has completed.

    Attributes
    ----------
    _GUARDED : set[str]
        Names of methods guarded by :class:`BuildGuardMixin`. By default
        ``{"forward", "_encode", "_decode"}``.
    """


    _GUARDED = {"forward", "_encode", "_decode"}
    
    def __init__(self) -> None:
        super().__init__()

    # --- gradient-enabled training APIs (to be implemented by subclasses) ---
    @abstractmethod
    def _encode(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> ModelOutput:
        """Encode a batch of inputs into a latent representation.

        Parameters
        ----------
        x : torch.Tensor
            Input batch to encode.
        *args
            Additional positional arguments passed to the encoder.
        **kwargs
            Additional keyword arguments passed to the encoder.

        Returns
        -------
        ModelOutput
            A model output object that must contain at least a latent code
            (for example an attribute named ``z``).
        """

        pass

    @abstractmethod
    def _decode(self, z: torch.Tensor, *args: Any, **kwargs: Any) -> ModelOutput:
        """Decode a batch of latent codes into reconstructed inputs.

        Parameters
        ----------
        z : torch.Tensor
            Latent representation batch to decode.
        *args
            Additional positional arguments passed to the decoder.
        **kwargs
            Additional keyword arguments passed to the decoder.

        Returns
        -------
        ModelOutput
            A model output object that must contain at least a reconstruction
            (for example an attribute named ``x_hat``).
        """

        pass

    @abstractmethod
    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> ModelOutput:
        """Full training forward pass of the autoencoder.

        This method is responsible for connecting the encoder and decoder and
        producing all outputs needed for training (for example latents,
        reconstructions and any auxiliary losses).

        Parameters
        ----------
        x : torch.Tensor
            Input batch to encode and decode.
        *args
            Additional positional arguments used by the subclass implementation.
        **kwargs
            Additional keyword arguments used by the subclass implementation.

        Returns
        -------
        ModelOutput
            A model output object that typically includes both the latent code
            (for example ``z``) and the reconstruction (for example ``x_hat``),
            plus any other training-specific quantities.
        """

        pass

    # --- inference-only convenience wrappers (no grad; optional eval mode) ---
    @torch.inference_mode()
    def encode(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> ModelOutput:
        """Encode inputs without tracking gradients.

        This is a thin wrapper around :meth:`_encode` executed under
        :func:`torch.inference_mode`, making it suitable for evaluation-time
        encoding.

        Parameters
        ----------
        x : torch.Tensor
            Input batch to encode.
        *args
            Additional positional arguments passed to :meth:`_encode`.
        **kwargs
            Additional keyword arguments passed to :meth:`_encode`.

        Returns
        -------
        ModelOutput
            The encoder :class:`ModelOutput`, typically containing at least a
            latent code (for example ``z``).
        """

        return self._encode(x, *args, **kwargs)

    @torch.inference_mode()
    def decode(self, z: torch.Tensor, *args: Any, **kwargs: Any) -> ModelOutput:
        """Decode latent codes without tracking gradients.

        This is a thin wrapper around :meth:`_decode` executed under
        :func:`torch.inference_mode`, making it suitable for evaluation-time
        decoding.

        Parameters
        ----------
        z : torch.Tensor
            Latent representation batch to decode.
        *args
            Additional positional arguments passed to :meth:`_decode`.
        **kwargs
            Additional keyword arguments passed to :meth:`_decode`.

        Returns
        -------
        ModelOutput
            The decoder :class:`ModelOutput`, typically containing at least a
            reconstruction (for example ``x_hat``).
        """

        return self._decode(z, *args, **kwargs)
    
    # ----- required build step -----
    @abstractmethod
    def build(self, input_sample: torch.Tensor) -> None:
        """Build the module given a representative input sample.

        Subclasses must implement this method to create any size-dependent layers,
        parameters or buffers (for example layers whose dimensions depend on
        ``input_sample.shape``). The implementation **must** set
        ``self._built = True`` once the module is fully initialized, otherwise
        :class:`NotBuiltError` will be raised by the build guard.

        This method is executed under ``torch.no_grad()`` by
        :class:`BuildGuardMixin`.

        Parameters
        ----------
        input_sample : torch.Tensor
            A representative input tensor used to infer sizes and initialize
            internal components.
        
        """

        pass

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        """Load a state dictionary into a built autoencoder.

        This override enforces that :meth:`build` has been called before loading
        weights, so that all parameters and buffers already exist with the correct
        shapes. If the model is not built yet, :class:`NotBuiltError` is raised
        with a hint to call :meth:`build` using a representative input.

        Parameters
        ----------
        state_dict : Mapping[str, Any]
            A state dictionary as produced by :meth:`state_dict`.
        strict : bool, optional
            If ``True`` (default), the keys in ``state_dict`` must exactly match
            the keys returned by :meth:`state_dict`. If ``False``, missing keys
            and unexpected keys are ignored.
        assign : bool, optional
            If ``True``, the incoming tensors in ``state_dict`` are directly
            assigned to the module's parameters and buffers, instead of copying
            data into existing tensors.

        Returns
        -------
        Any
            The same value returned by :meth:`torch.nn.Module.load_state_dict`,
            typically a :class:`torch.nn.modules.module._IncompatibleKeys` object.
        """

        if not self._built:
            raise NotBuiltError(
                "load_state_dict called before build(). "
                "Call model.build(example_x) so parameters exist, then load."
            )
        return super().load_state_dict(state_dict=state_dict, strict=strict, assign=assign)
    
    @abstractmethod
    def compute_loss(self, x: torch.Tensor, output: ModelOutput, *args: Any, **kwargs: Any) -> LossResult:
        """Compute the loss for the autoencoder.

        This abstract method must be implemented by subclasses to compute
        the appropriate loss objective for the model. Subclasses may support
        additional hyperparameters and configuration options.

        Parameters
        ----------
        x : torch.Tensor
            Ground-truth inputs of shape ``[B, ...]``.
        model_output : ModelOutput
            Output from the forward pass, containing reconstructions, latent codes,
            and any other information needed to compute the loss.
        *args
            Additional positional arguments (subclass-specific).
        **kwargs
            Additional keyword arguments (subclass-specific).

        Returns
        -------
        LossResult
            Result containing the loss objective and optional diagnostics.
        """
        pass
