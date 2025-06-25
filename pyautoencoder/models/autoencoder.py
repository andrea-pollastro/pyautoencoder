from typing import Tuple
from utils.loss import log_likelihood
import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    """
    A simple Autoencoder model.

    This class encapsulates an autoencoder composed of a user-defined encoder and decoder.
    The encoder maps the input to a latent representation, and the decoder reconstructs the input from the latent space.

    Args:
        encoder (nn.Module): A neural network that encodes the input into a latent representation.
        decoder (nn.Module): A neural network that decodes the latent representation back to the input space.

    Methods:
        forward(x): Computes the reconstructed input and latent representation.
        encode(x): Returns the latent representation without computing gradients (inference mode) and in eval mode.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - x_hat: The reconstructed input.
            - z: The latent representation.
    """
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z
    
    @torch.inference_mode()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.encoder(x)
