from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class BaseVAE(nn.Module, ABC):
    """
    Abstract base class for Variational Autoencoders (VAEs).

    This class provides the shared structure and behavior for VAE models, including:
    - encoder and decoder modules,
    - layers for computing latent mean and log-variance,
    - the reparameterization trick for sampling latent variables.

    Subclasses must implement the `forward` method.

    Args:
        encoder (nn.Module): Neural network encoder that outputs a feature representation.
        decoder (nn.Module): Neural network decoder that reconstructs input from latent space.
        latent_dim (int): Dimensionality of the latent variable z.
    """
    def __init__(self, 
                 encoder: nn.Module, 
                 decoder: nn.Module, 
                 latent_dim: int):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.fc_mu = nn.LazyLinear(latent_dim)
        self.fc_logvar = nn.LazyLinear(latent_dim)

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
