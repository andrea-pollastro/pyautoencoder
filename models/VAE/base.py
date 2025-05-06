from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class BaseVAE(nn.Module, ABC):
    """
    Abstract base class for Variational Autoencoders (VAEs).

    This class defines the common interface and utility for VAE-style models.
    It provides:
    - A shared reparameterization method,
    - An abstract forward method to be implemented by subclasses.

    Subclasses should implement the full model architecture and define
    how the forward pass returns the reconstructed input and latent statistics.
    """
    def __init__(self):
        super().__init__()

    def reparametrize(self, 
                      mu: torch.Tensor, 
                      log_var: torch.Tensor, 
                      L: int = 1) -> torch.Tensor:
        """
        Applies the reparameterization trick to sample latent variables z ~ N(mu, exp(log_var)).

        Args:
            mu (torch.Tensor): Mean tensor of shape [B, latent_dim].
            log_var (torch.Tensor): Log-variance tensor of shape [B, latent_dim].
            L (int): Number of samples per input in the latent space.

        Returns:
            torch.Tensor: Sampled latent variables z of shape [B, L, latent_dim].
        """
        if self.training:
            std = torch.exp(0.5 * log_var)
            mu = mu.unsqueeze(1).expand(-1, L, -1)
            std = std.unsqueeze(1).expand(-1, L, -1)
            eps = torch.randn_like(std)
            z = mu + std * eps
        else:
            z = mu.unsqueeze(1).expand(-1, L, -1)
            
        return z

    @abstractmethod
    def forward(self, x: torch.Tensor, L: int = 1):
        """
        Forward pass of the VAE model to be implemented by subclasses.

        Args:
            x (torch.Tensor): Input tensor.
            L (int): Number of samples for the reparameterization (used for IWAE or Monte Carlo).

        Returns:
            Depends on subclass implementation.
        """
        pass
