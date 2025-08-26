from typing import Tuple
import torch
import torch.nn as nn
from ..base import BaseVariationalAutoencoder
from .stochastic_layers import FullyFactorizedGaussian

class VAE(BaseVariationalAutoencoder):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        latent_dim: int,
    ):
        """
        Standard Variational Autoencoder (VAE) with a single latent layer.
        Implementation following "Kingma, D. P., & Welling, M. (2013). Auto-encoding variational bayes."

        Components:
            - encoder: x → features f(x)  (shape [B, F])
            - decoder: z → x_hat

        Args:
            encoder (nn.Module): Network mapping input x to feature vector f(x), shape [B, F].
            decoder (nn.Module): Network mapping latent z to reconstruction x_hat.
            latent_dim (int): Dimensionality of the latent space (D_z).
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.sampling_layer = FullyFactorizedGaussian(latent_dim=latent_dim)

    def _encode(self, x: torch.Tensor, S: int = 1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encodes inputs and draws latent samples via the sampling layer.

        Args:
            x (torch.Tensor): Inputs, shape [B, ...]. The encoder must output a flat feature
                              vector per sample (e.g., [B, F]) compatible with the sampling head.
            S (int): Number of Monte Carlo samples per input.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - z       (torch.Tensor): Latent samples, shape [B, S, D_z].
                - mu      (torch.Tensor): Mean of q(z|x), shape [B, D_z].
                - log_var (torch.Tensor): Log-variance of q(z|x), shape [B, D_z].

        Notes:
            - The sampling layer follows the module's training mode by default
              (samples in train mode; tiles μ in eval).
        """
        f = self.encoder(x)                      # [B, F]
        z, mu, log_var = self.sampling_layer(f, S=S)  # z: [B, S, D_z]
        return z, mu, log_var

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decodes latent variables to reconstruction logits (or means).

        Args:
            z (torch.Tensor): Latent inputs, shape **[B, S, D_z]** (S is kept even when S=1).

        Returns:
            torch.Tensor: Reconstructions x_hat, shape **[B, S, ...]**.

        Notes:
            - This method reshapes internally: flattens the (B·S) batch, applies the decoder,
              then restores [B, S, ...]. No additional reshaping is needed in `forward`.
        """
        if z.dim() != 3:
            raise ValueError(f"Expected z with shape [B, S, D_z]; got {tuple(z.shape)}")
        B, S, D_z = z.shape
        x_hat = self.decoder(z.reshape(B * S, D_z))   # [B*S, ...]
        return x_hat.view(B, S, *x_hat.shape[1:])     # [B, S, ...]

    def forward(self, x: torch.Tensor, S: int = 1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Runs the VAE pipeline: encode → (sample S) → decode.

        Args:
            x (torch.Tensor): Inputs, shape [B, ...].
            S (int): Number of latent samples per input for Monte Carlo estimates.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - x_hat  (torch.Tensor): Reconstructions/logits,  shape [B, S, ...].
                - z      (torch.Tensor): Latent samples,          shape [B, S, D_z].
                - mu     (torch.Tensor): Mean of q(z|x),          shape [B, D_z].
                - log_var(torch.Tensor): Log-variance of q(z|x),  shape [B, D_z].

        Notes:
            - When S > 1, broadcasting x → [B, S, ...] during loss computation allows
              evaluating log p(x | z_s) for each sample without copying x.
            - For Bernoulli likelihoods, ensure the decoder outputs logits.
        """
        z, mu, log_var = self._encode(x, S=S)  # z: [B, S, D_z]
        x_hat = self._decode(z)                # [B, S, ...]
        return x_hat, z, mu, log_var
