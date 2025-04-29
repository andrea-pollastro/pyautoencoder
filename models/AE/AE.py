from typing import Tuple
import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super(Autoencoder, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z
