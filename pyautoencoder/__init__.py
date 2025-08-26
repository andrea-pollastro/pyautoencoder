"""PyAutoencoder: A clean, modular PyTorch library for autoencoder models."""

from .models.autoencoder import AE
from .models.variational.vae import VAE
from .loss.wrapper import AutoencoderLoss, VAELoss, LossComponents

__version__ = "0.1.0"

__all__ = [
    # Models
    'AE',
    'VAE',
    
    # Losses
    'AutoencoderLoss',
    'VAELoss',
    'LossComponents'
]