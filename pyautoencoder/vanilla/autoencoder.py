import torch
import torch.nn as nn
from dataclasses import dataclass

from .._base.base import BaseAutoencoder, ModelOutput

@dataclass(slots=True, repr=False)
class AEEncodeOutput(ModelOutput):
    """Output of the Autoencoder encoder stage.

    Attributes
    ----------
    z : torch.Tensor
        Latent code of shape ``[B, ...]`` produced by :meth:`AE._encode`
        or :meth:`AE.encode`.
    """

    z: torch.Tensor

@dataclass(slots=True, repr=False)
class AEDecodeOutput(ModelOutput):
    """Output of the Autoencoder decoder stage.

    Attributes
    ----------
    x_hat : torch.Tensor
        Reconstruction (or logits) of shape ``[B, ...]`` produced by
        :meth:`AE._decode` or :meth:`AE.decode`.
    """

    x_hat: torch.Tensor

@dataclass(slots=True, repr=False)
class AEOutput(ModelOutput):
    """Output of the full Autoencoder forward pass.

    Attributes
    ----------
    x_hat : torch.Tensor
        Reconstruction (or logits) of shape ``[B, ...]``.
    z : torch.Tensor
        Latent code of shape ``[B, ...]``.
    """

    x_hat: torch.Tensor
    z: torch.Tensor

class AE(BaseAutoencoder):
    """Standard Autoencoder composed of a user-defined encoder and decoder.

    The model follows the :class:`BaseAutoencoder` interface and implements:

    * ``_encode(x)`` – maps inputs ``x`` to latent codes ``z``.
    * ``_decode(z)`` – maps latent codes ``z`` to reconstructions ``x_hat``.
    * ``forward(x)`` – full training forward pass returning both ``z`` and ``x_hat``.

    The encoder and decoder are arbitrary :class:`torch.nn.Module` instances.
    """

    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        """Construct an Autoencoder from an encoder and decoder module.

        Parameters
        ----------
        encoder : nn.Module
            Module implementing the mapping ``x → z``.
        decoder : nn.Module
            Module implementing the mapping ``z → x_hat``.
        """

        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    # --- training-time hooks required by BaseAutoencoder ---
    def _encode(self, x: torch.Tensor) -> AEEncodeOutput:
        """Encode inputs into latent representations.

        Parameters
        ----------
        x : torch.Tensor
            Input batch of shape ``[B, ...]``.

        Returns
        -------
        AEEncodeOutput
            Dataclass containing the latent code ``z``.
        """

        z = self.encoder(x)
        return AEEncodeOutput(z=z)

    def _decode(self, z: torch.Tensor) -> AEDecodeOutput:
        """Decode latent variables into reconstructions.

        Parameters
        ----------
        z : torch.Tensor
            Latent batch of shape ``[B, ...]``.

        Returns
        -------
        AEDecodeOutput
            Dataclass containing the reconstruction ``x_hat``.
        """

        x_hat = self.decoder(z)
        return AEDecodeOutput(x_hat=x_hat)

    def forward(self, x: torch.Tensor) -> AEOutput:
        """Full training forward pass with gradients.

        Parameters
        ----------
        x : torch.Tensor
            Input batch of shape ``[B, ...]``.

        Returns
        -------
        AEOutput
            Dataclass containing both the reconstruction ``x_hat`` and the
            latent code ``z``.
        """

        enc = self._encode(x)      # AEEncodeOutput(z)
        dec = self._decode(enc.z)  # AEDecodeOutput(x_hat)
        return AEOutput(x_hat=dec.x_hat, z=enc.z)
    
    def build(self, input_sample: torch.Tensor) -> None:
        """Build the Autoencoder using a representative input.

        For standard Autoencoders with fixed-size encoder and decoder modules,
        no size-dependent initialization is required, so this method simply sets
        ``self._built = True``.

        Parameters
        ----------
        input_sample : torch.Tensor
            Representative input tensor (ignored).
        """

        self._built = True
