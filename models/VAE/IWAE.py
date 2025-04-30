from utils.loss import IWAE_ELBO
from .VAE import VariationalAutoencoder
import torch
import torch.nn as nn

class ImportanceWeightedAutoencoder(VariationalAutoencoder):
    def __init__(self, 
                 encoder: nn.Module, 
                 decoder: nn.Module, 
                 latent_dim: int):
        super().__init__(encoder=encoder, decoder=decoder, latent_dim=latent_dim)
        
        self.elbo = IWAE_ELBO

    def forward(self, x: torch.Tensor, L: int = 2):
        if self.training and L < 2:
            raise ValueError("L must be greater than or equal to 2 for IWAE.")
        return super().forward(x, L=L)
