import torch
import torch.nn as nn
from dataclasses import dataclass

from .base import BaseAutoencoder, ModelOutput

@dataclass(slots=True, repr=False)
class AEEncodeOutput(ModelOutput):
    """Output of AE._encode / AE.encode.

    Attributes:
        z (torch.Tensor): Latent code, shape [B, ...].
    """
    z: torch.Tensor

@dataclass(slots=True, repr=False)
class AEDecodeOutput(ModelOutput):
    """Output of AE._decode / AE.decode.

    Attributes:
        x_hat (torch.Tensor): Reconstruction/logits, shape [B, ...].
    """
    x_hat: torch.Tensor

@dataclass(slots=True, repr=False)
class AEOutput(ModelOutput):
    """Output of AE.forward.

    Attributes:
        x_hat (torch.Tensor): Reconstruction/logits, shape [B, ...].
        z     (torch.Tensor): Latent code,           shape [B, ...].
    """
    x_hat: torch.Tensor
    z: torch.Tensor

class AE(BaseAutoencoder):
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        """
        A simple Autoencoder model.

        Wraps a user-defined encoder and decoder:
          - encoder: x -> z
          - decoder: z -> x_hat
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    # --- training-time hooks required by BaseAutoencoder ---
    def _encode(self, x: torch.Tensor) -> AEEncodeOutput:
        """Compute latent representation.

        Args:
            x: Input tensor, shape [B, ...].

        Returns:
            AEEncodeOutput: with field `z` of shape [B, ...].
        """
        z = self.encoder(x)
        return AEEncodeOutput(z=z)

    def _decode(self, z: torch.Tensor) -> AEDecodeOutput:
        """Decode latent to reconstruction/logits.

        Args:
            z: Latent tensor, shape [B, ...].

        Returns:
            AEDecodeOutput: with field `x_hat` of shape [B, ...].
        """
        x_hat = self.decoder(z)
        return AEDecodeOutput(x_hat=x_hat)

    def forward(self, x: torch.Tensor) -> AEOutput:
        """Training forward pass with gradients.

        Args:
            x: Input tensor, shape [B, ...].

        Returns:
            AEOutput with:
                - x_hat: Reconstruction/logits, shape [B, ...]
                - z:     Latent code,           shape [B, ...]
        """
        enc = self._encode(x)      # AEEncodeOutput(z)
        dec = self._decode(enc.z)  # AEDecodeOutput(x_hat)
        return AEOutput(x_hat=dec.x_hat, z=enc.z)
    
    def build(self, input_sample: torch.Tensor) -> None:
        self._build = True
