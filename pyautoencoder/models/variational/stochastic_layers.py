import torch
import torch.nn as nn

class FullyFactorizedGaussian(nn.Module):
    """
    Head that maps features to a fully factorized Gaussian posterior q(z|x)
    with parameters (mu, log_var), and (optionally) samples z via the
    reparameterization trick.

    Args:
        latent_dim (int): dimensionality of z.

    Input:
        x: [B, F]  (F can be inferred on first forward thanks to LazyLinear)

    Returns:
        z:       [B, S, latent_dim], sampled in training only
        mu:      [B, latent_dim]
        log_var: [B, latent_dim]
    """
    def __init__(self, latent_dim: int):
        super().__init__()
        self.mu = nn.LazyLinear(latent_dim)
        self.log_var = nn.LazyLinear(latent_dim)

    def forward(self, x: torch.Tensor, S: int = 1):
        mu = self.mu(x)            # [B, Dz]
        log_var = self.log_var(x)  # [B, Dz]

        if self.training:
            std = torch.exp(0.5 * log_var)              # [B, Dz]
            mu_e  = mu.unsqueeze(1).expand(-1, S, -1)   # [B, S, Dz]
            std_e = std.unsqueeze(1).expand(-1, S, -1)  # [B, S, Dz]
            eps = torch.randn_like(std_e)
            z = mu_e + std_e * eps                      # [B, S, Dz]
        else:
            z = mu.unsqueeze(1).expand(-1, S, -1)       # [B, S, Dz]

        return z, mu, log_var
