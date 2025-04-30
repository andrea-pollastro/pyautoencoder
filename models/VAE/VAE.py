from .base import BaseVAE
from utils.loss import ELBO
from typing import Tuple
import torch
import torch.nn as nn

class VariationalAutoencoder(BaseVAE):
    """
    Variational Autoencoder (VAE) implementation using the reparameterization trick.

    Args:
        encoder (nn.Module): Neural network encoder producing feature representations.
        decoder (nn.Module): Neural network decoder reconstructing inputs from latent variables.
        latent_dim (int): Dimensionality of the latent space.

    Forward Args:
        x (torch.Tensor): Input tensor of shape [B, ...].
        L (int): Number of latent samples per input (default: 1).

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            - x_hat (torch.Tensor): Reconstructed inputs of shape [B, L, ...].
            - z (torch.Tensor): Sampled latent variables of shape [B, L, latent_dim].
            - mu (torch.Tensor): Mean of q(z|x), shape [B, latent_dim].
            - log_var (torch.Tensor): Log-variance of q(z|x), shape [B, latent_dim].
    """
    def __init__(self, 
                 encoder: nn.Module, 
                 decoder: nn.Module, 
                 latent_dim: int):
        super().__init__(encoder=encoder, decoder=decoder, latent_dim=latent_dim)
        
        self.elbo = ELBO

    def forward(self, 
                x: torch.Tensor, 
                L: int = 1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B = x.size(0)

        # q(z|x)
        x_f = self.encoder(x).flatten(1)
        mu = self.fc_mu(x_f)
        log_var = self.fc_logvar(x_f)

        # z ~ q(z|x)
        z = self.reparametrize(mu=mu, log_var=log_var, L=L)

        # p(z|x)
        z_flat = z.reshape(B * L, -1)
        x_hat = self.decoder(z_flat)
        x_hat = x_hat.view(B, L, *x.shape[1:])

        return x_hat, z, mu, log_var