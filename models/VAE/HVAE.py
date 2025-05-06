from .base import BaseVAE
import torch
import torch.nn as nn
from typing import List, Tuple
from utils.loss import hierarchical_ELBO

class HierarchicalVAE(BaseVAE):
    """
    Hierarchical Variational Autoencoder (HVAE) with configurable number of latent layers.

    This model defines a generative hierarchy where each latent variable z_l is conditioned
    on the variable above it (z_{l+1}), and the inference model encodes bottom-up from x.

    Args:
        encoders (List[nn.Module]): List of encoder modules, one for each latent layer.
                                    The first encoder takes x as input, subsequent ones take z_{l-1}.
        decoders (List[nn.Module]): List of decoder modules, one per latent layer in reverse order.
                                    The last decoder reconstructs x from z_1.
        latent_dims (List[int]): List of dimensionalities for each latent variable z_l.
    """
    def __init__(self,
                 encoders: List[nn.Module],
                 decoders: List[nn.Module],
                 latent_dims: List[int]):
        super().__init__()

        assert len(encoders) == len(decoders) == len(latent_dims), \
            "Mismatch in number of encoders, decoders, or latent_dims"

        self.num_layers = len(latent_dims)
        self.latent_dims = latent_dims
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(decoders)
        self.fc_mus = nn.ModuleList([nn.LazyLinear(dim) for dim in latent_dims])
        self.fc_logvars = nn.ModuleList([nn.LazyLinear(dim) for dim in latent_dims])
        
        self.elbo = hierarchical_ELBO

    def forward(self, 
                x: torch.Tensor, 
                L: int = 1) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass through the hierarchical VAE.

        Performs bottom-up encoding through multiple latent layers, applies the reparameterization
        trick at each level, and then decodes top-down to reconstruct the input.

        Args:
            x (torch.Tensor): Input tensor of shape [B, ...].
            L (int): Number of samples per input for Monte Carlo approximation.

        Returns:
            Tuple containing:
                - x_hat (torch.Tensor): Reconstructed inputs, shape [B, L, ...].
                - zs (List[torch.Tensor]): List of sampled latent variables, each of shape [B, L, D_l].
                - mus (List[torch.Tensor]): List of means for q(z_l|·), each of shape [B, D_l].
                - logvars (List[torch.Tensor]): List of log-variances for q(z_l|·), each of shape [B, D_l].
        """
        B = x.size(0)
        zs, mus, logvars = [], [], []

        current_input = x
        for i in range(self.num_layers):
            h = self.encoders[i](current_input).flatten(1)
            mu = self.fc_mus[i](h)
            logvar = self.fc_logvars[i](h)
            z = self.reparametrize(mu, logvar, L=L)
            current_input = z.mean(dim=1)

            zs.append(z)
            mus.append(mu)
            logvars.append(logvar)

        current_z = zs[-1]
        for i in reversed(range(self.num_layers)):
            z_flat = current_z.reshape(B * L, -1)
            out = self.decoders[i](z_flat)
            if i > 0:
                current_z = out.view(B, L, -1)
            else:
                x_hat = out.view(B, L, *x.shape[1:])

        return x_hat, zs, mus, logvars
