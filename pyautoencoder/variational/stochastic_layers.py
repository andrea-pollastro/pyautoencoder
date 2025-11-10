import torch
import torch.nn as nn
from typing import Optional

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
        self.latent_dim = latent_dim
        self.mu: Optional[nn.Linear] = None
        self.log_var: Optional[nn.Linear] = None
        self._built = False

    def build(self, input_sample: torch.Tensor) -> None:
        """
        Initialize layers using a sample input tensor `x` of shape [B, F].
        Idempotent if called again with the same F.
        """
        if not isinstance(input_sample, torch.Tensor):
            raise TypeError("build(x) expects a torch.Tensor.")
        if input_sample.ndim != 2:
            raise ValueError(f"build(x): expected shape [B, F], got {tuple(input_sample.shape)}. Flatten upstream.")
        if input_sample.shape[1] <= 0:
            raise ValueError("build(x): F (feature dimension) must be > 0.")
        
        in_features = int(input_sample.shape[1])

        self.mu = nn.Linear(in_features, self.latent_dim)
        self.log_var = nn.Linear(in_features, self.latent_dim)
        self.to(device=input_sample.device, dtype=input_sample.dtype)

        self.in_features = in_features
        self._built = True

    def forward(self, x: torch.Tensor, S: int = 1):
        if not self._built:
            raise RuntimeError("FullyFactorizedGaussian not built. Call `.build(x)` first.")

        mu = self.mu(x)            # type: ignore       # [B, Dz]
        log_var = self.log_var(x)  # type: ignore       # [B, Dz]

        if self.training:
            std = torch.exp(0.5 * log_var)              # [B, Dz]
            mu_e  = mu.unsqueeze(1).expand(-1, S, -1)   # [B, S, Dz]
            std_e = std.unsqueeze(1).expand(-1, S, -1)  # [B, S, Dz]
            eps = torch.randn_like(std_e)
            z = mu_e + std_e * eps                      # [B, S, Dz]
        else:
            z = mu.unsqueeze(1).expand(-1, S, -1)       # [B, S, Dz]

        return z, mu, log_var
    
    @property
    def built(self) -> bool:
        return self._built