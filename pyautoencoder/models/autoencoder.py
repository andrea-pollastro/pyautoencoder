from typing import Tuple
import torch
import torch.nn as nn
from .base import BaseAutoencoder

class AE(BaseAutoencoder):
    """
    A simple Autoencoder model.

    This class wraps a user-defined encoder and decoder.
    The encoder maps the input x to a latent representation z, and the decoder
    reconstructs x_hat from z.

    Args:
        encoder (nn.Module): Network mapping x -> z.
        decoder (nn.Module): Network mapping z -> x_hat.

    Methods:
        forward(x): Training forward with gradients; returns (x_hat, z).
        encode(x, use_eval=True): Inference wrapper (no grad, optional eval()) inherited from BaseAutoencoder.
        decode(z, use_eval=True): Inference wrapper (no grad, optional eval()) inherited from BaseAutoencoder.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - x_hat: Reconstruction/logits, shape [B, ...].
            - z:     Latent code,        shape [B, D_z] (or similar).
    """
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def _encode(self, x: torch.Tensor) -> torch.Tensor: return self.encoder(x)
    def _decode(self, z: torch.Tensor) -> torch.Tensor: return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self._encode(x)
        x_hat = self._decode(z)
        return x_hat, z
