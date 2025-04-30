from utils.loss import IWAE_ELBO
from .VAE import VariationalAutoencoder
import torch
import torch.nn as nn

class ImportanceWeightedAutoencoder(VariationalAutoencoder):
    """
    Importance Weighted Autoencoder (IWAE) implementation.

    Extends the standard VariationalAutoencoder by replacing the ELBO with the
    IWAE objective, providing a tighter lower bound on the marginal log-likelihood.

    Args:
        encoder (nn.Module): Neural network encoder producing latent parameters.
        decoder (nn.Module): Neural network decoder reconstructing inputs from latents.
        latent_dim (int): Dimensionality of the latent space.

    Forward Args:
        x (torch.Tensor): Input tensor.
        L (int): Number of importance samples. Must be >= 2 during training.

    Returns:
        Depends on the implementation of the base VariationalAutoencoder.
        Typically includes the reconstruction, latent samples, and loss terms.
    
    Raises:
        ValueError: If L < 2 during training.
    """
    def __init__(self, 
                 encoder: nn.Module, 
                 decoder: nn.Module, 
                 latent_dim: int):
        super().__init__(encoder=encoder, decoder=decoder, latent_dim=latent_dim)
        
        self.elbo = IWAE_ELBO

    def forward(self, x: torch.Tensor, L: int = 2):
        if self.training and L < 2:
            raise ValueError("L must be greater than or equal to 2 for IWAE.")
        return super().forward(x, L=L)
