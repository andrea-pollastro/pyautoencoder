from .stochastic_layers import FullyFactorizedGaussian
from typing import Tuple
import torch
import torch.nn as nn

class VariationalAutoencoder(nn.Module):
    """
    Standard Variational Autoencoder (VAE) implementation using the reparameterization trick.

    This model assumes a single latent layer and consists of:
    - an encoder producing parameters of the approximate posterior q(z|x),
    - a decoder reconstructing the input from latent samples z.

    Args:
        encoder (nn.Module): Encoder network mapping input x to a latent representation.
        decoder (nn.Module): Decoder network reconstructing x from latent variable z.
        latent_dim (int): Dimensionality of the latent space.
    """
    def __init__(self, 
                 encoder: nn.Module, 
                 decoder: nn.Module, 
                 latent_dim: int,
                 sampling_layer: str = 'fully_factorized_gaussian'):
        super().__init__()

        self.latent_dim = latent_dim
        self.encoder = encoder
        self.decoder = decoder
        if sampling_layer == 'fully_factorized_gaussian':
            self.sampling_layer = FullyFactorizedGaussian(latent_dim=latent_dim)
        else:
            raise ValueError(f'Sampling layer {sampling_layer} not available.')
        
    def forward(self, 
                x: torch.Tensor, 
                L: int = 1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the VAE.

        Encodes the input to obtain parameters of q(z|x), samples latent variables z using
        the reparameterization trick, and decodes z to reconstruct the input.

        Args:
            x (torch.Tensor): Input tensor of shape [B, ...], where B is the batch size.
            L (int): Number of samples per input for Monte Carlo estimates (default: 1).

        Returns:
            Tuple containing:
                - x_hat (torch.Tensor): Reconstructed inputs, shape [B, L, ...].
                - z (torch.Tensor): Sampled latent variables, shape [B, L, latent_dim].
                - mu (torch.Tensor): Mean of q(z|x), shape [B, latent_dim].
                - log_var (torch.Tensor): Log-variance of q(z|x), shape [B, latent_dim].
        """
        B = x.size(0)

        # z ~ q(z|x)
        x_f = self.encoder(x)
        z, mu, log_var = self.sampling_layer(x=x_f, L=L)

        # p(x|z)
        z_flat = z.reshape(B * L, -1)
        x_hat = self.decoder(z_flat)
        x_hat = x_hat.view(B, L, *x.shape[1:])

        return x_hat, z, mu, log_var
