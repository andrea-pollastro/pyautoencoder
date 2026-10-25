from typing import Any
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields

from ..loss.base import LossResult

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
    
class BaseAutoencoder(nn.Module, ABC):
    """Base class for autoencoders.

    This class defines a common interface for autoencoders that split their
    logic into *encoding*, *decoding* and *forward* passes.

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

    Build step
    ----------
    :meth:`build` performs a no-grad forward pass with a representative input
    to materialize any lazy (size-inferred) layers. Call it once before
    training, e.g. before creating the optimizer or applying a custom weight
    initialization. Loading a state dict does not require it: PyTorch
    materializes lazy layers from the loaded tensors.
    """

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

    @torch.no_grad()
    def build(self, input_sample: torch.Tensor) -> None:
        """Materialize lazy layers with a representative input.

        Performs a no-grad forward pass so that any
        :class:`~torch.nn.modules.lazy.LazyModuleMixin` layers infer their
        shapes. Call this once before training, e.g. before creating the
        optimizer or applying a custom weight initialization.

        The pass runs in eval mode, so it does not update BatchNorm running
        statistics and does not apply dropout or latent sampling. The
        train/eval mode of every submodule is restored afterwards.

        Parameters
        ----------
        input_sample : torch.Tensor
            A representative input batch (e.g., a single batch from the
            training set). Only its shape matters.
        """
        modes = {m: m.training for m in self.modules()}
        self.eval()
        try:
            self(input_sample)
        finally:
            for m, training in modes.items():
                m.training = training
    
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
        output : ModelOutput
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
