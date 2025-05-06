from utils.loss import IWAE_ELBO
from .VAE import VariationalAutoencoder
import torch
import torch.nn as nn

class ImportanceWeightedAutoencoder(VariationalAutoencoder):
    """
    Importance Weighted Autoencoder (IWAE).

    Extends the standard VAE by using the IWAE objective, which provides a tighter lower bound
    on the marginal log-likelihood by averaging over multiple importance-weighted samples.

    Args:
        encoder (nn.Module): Encoder network that outputs latent parameters (mu, log_var).
        decoder (nn.Module): Decoder network reconstructing inputs from latent variables.
        latent_dim (int): Dimensionality of the latent space.
    """
    def __init__(self, 
                 encoder: nn.Module, 
                 decoder: nn.Module, 
                 latent_dim: int):
        super().__init__(encoder=encoder, decoder=decoder, latent_dim=latent_dim)
        
        self.elbo = IWAE_ELBO

    def forward(self, x: torch.Tensor, L: int = 2):
        """
        Forward pass of the IWAE model.

        Computes the reconstruction and latent representations using multiple samples per input,
        as required by the IWAE objective.

        Args:
            x (torch.Tensor): Input tensor of shape [B, ...].
            L (int): Number of importance-weighted samples (must be >= 2 during training).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - x_hat (torch.Tensor): Reconstructed inputs, shape [B, L, ...].
                - z (torch.Tensor): Sampled latent variables, shape [B, L, latent_dim].
                - mu (torch.Tensor): Mean of q(z|x), shape [B, latent_dim].
                - log_var (torch.Tensor): Log-variance of q(z|x), shape [B, latent_dim].

        Raises:
            ValueError: If L < 2 while in training mode.
        """
        if self.training and L < 2:
            raise ValueError("L must be greater than or equal to 2 for IWAE training.")
        return super().forward(x, L=L)
