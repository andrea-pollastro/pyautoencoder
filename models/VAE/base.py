from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class BaseVAE(nn.Module, ABC):
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
        if self.training:
            std = torch.sqrt(torch.exp(log_var))
            mu = mu.unsqueeze(1).expand(-1, L, -1)
            std = std.unsqueeze(1).expand(-1, L, -1)
            eps = torch.randn_like(std)
            z = mu + std * eps
        else:
            z = mu
            
        return z

    @abstractmethod
    def forward(self, x: torch.Tensor, L: int = 1):
        pass
