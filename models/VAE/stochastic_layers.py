import torch
import torch.nn as nn

class IsotropicGaussian(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.mu = nn.LazyLinear(latent_dim)
        self.log_var = nn.LazyLinear(latent_dim)

    def forward(self, x: torch.Tensor, L: int = 1):
        """
        Applies the reparameterization trick to sample latent variables z ~ N(mu, exp(log_var)).

        Args:
            x (torch.Tensor): Input tensor of shape [B, ...], where B is the batch size.
            L (int): Number of samples per input in the latent space.

        Returns:
            torch.Tensor: Sampled latent variables z of shape [B, L, latent_dim].
        """
        mu = self.mu(x)
        log_var = self.log_var(x)
        
        if self.training:
            std = torch.exp(0.5 * log_var)
            mu = mu.unsqueeze(1).expand(-1, L, -1)
            std = std.unsqueeze(1).expand(-1, L, -1)
            eps = torch.randn_like(std)
            z = mu + std * eps
        else:
            z = mu.unsqueeze(1).expand(-1, L, -1)
            
        return z, mu, log_var