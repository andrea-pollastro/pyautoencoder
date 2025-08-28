import torch
import torch.nn as nn
from dataclasses import dataclass

from .._base.base import BaseAutoencoder, ModelOutput
from .stochastic_layers import FullyFactorizedGaussian

@dataclass(slots=True, repr=False)
class VAEEncodeOutput(ModelOutput):
    """Output of VAE._encode / VAE.encode.

    Attributes:
        z       (torch.Tensor): Latent samples, shape [B, S, D_z] (S can be 1).
        mu      (torch.Tensor): Mean of q(z|x), shape [B, D_z].
        log_var (torch.Tensor): Log-variance of q(z|x), shape [B, D_z].
    """
    z: torch.Tensor
    mu: torch.Tensor
    log_var: torch.Tensor

@dataclass(slots=True, repr=False)
class VAEDecodeOutput(ModelOutput):
    """Output of VAE._decode / VAE.decode.

    Attributes:
        x_hat (torch.Tensor): Reconstructions/logits, shape [B, S, ...] 
    """
    x_hat: torch.Tensor


@dataclass(slots=True, repr=False)
class VAEOutput(ModelOutput):
    """Output of VAE.forward.

    Attributes:
        x_hat   (torch.Tensor): Reconstructions/logits,  shape [B, S, ...].
        z       (torch.Tensor): Latent samples,          shape [B, S, D_z].
        mu      (torch.Tensor): Mean of q(z|x),          shape [B, D_z].
        log_var (torch.Tensor): Log-variance of q(z|x),  shape [B, D_z].
    """
    x_hat: torch.Tensor
    z: torch.Tensor
    mu: torch.Tensor
    log_var: torch.Tensor

class VAE(BaseAutoencoder):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        latent_dim: int,
    ):
        """
        Standard Variational Autoencoder (VAE) with a single latent layer.
        Follows Kingma & Welling (2013), "Auto-Encoding Variational Bayes".

        Components:
            - encoder: x → features extracted f(x) before sampling layer (shape [B, F])
            - sampling_layer: (μ, log σ², S) → z                         (shape [B, S, D_z])
            - decoder: z → x_hat

        Args:
            encoder (nn.Module): Maps input x to feature vector f(x), shape [B, F],
                                 internally producing μ and log σ² for q(z|x).
            decoder (nn.Module): Maps latent z to reconstruction x_hat.
            latent_dim (int):    Dimensionality of the latent space (D_z).
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.sampling_layer = FullyFactorizedGaussian(latent_dim=latent_dim)

    # --- training-time hooks required by BaseAutoencoder ---
    def _encode(self, x: torch.Tensor, S: int = 1) -> VAEEncodeOutput:
        """Encode inputs and draw S latent samples via the sampling layer.

        Args:
            x (torch.Tensor): Inputs, shape [B, ...]. The encoder must output a flat feature
                              vector per sample compatible with the sampling head.
            S (int): Number of Monte Carlo samples per input.

        Returns:
            VAEEncodeOutput with:
                - z:       [B, S, D_z]
                - mu:      [B, D_z]
                - log_var: [B, D_z]

        Notes:
            The sampling layer typically follows module training mode:
              - train(): sample from q(z|x)
              - eval():  tile μ (or use deterministic behavior)
        """
        f = self.encoder(x)
        z, mu, log_var = self.sampling_layer(f, S=S)
        return VAEEncodeOutput(z=z, mu=mu, log_var=log_var)

    def _decode(self, z: torch.Tensor) -> VAEDecodeOutput:
        """Decode latent variables to reconstruction logits (or means).

        Args:
            z (torch.Tensor): Latent inputs, shape [B, S, D_z].

        Returns:
            VAEDecodeOutput with:
                - x_hat: [B, S, ...]
        """
        B, S, D_z = z.shape
        x_hat_flat = self.decoder(z.reshape(B * S, D_z))  # [B * S, ...]
        x_hat = x_hat_flat.reshape(B, S, *x_hat_flat.shape[1:])
        return VAEDecodeOutput(x_hat=x_hat)

    def forward(self, x: torch.Tensor, S: int = 1) -> VAEOutput:
        """Full VAE pass: encode → sample S → decode.

        Args:
            x (torch.Tensor): Inputs, shape [B, ...].
            S (int): Number of latent samples per input for Monte Carlo estimates.

        Returns:
            VAEOutput with:
                - x_hat:  Reconstructions/logits,  shape [B, S, ...]
                - z:      Latent samples,          shape [B, S, D_z]
                - mu:     Mean of q(z|x),          shape [B, D_z]
                - log_var:Log-variance of q(z|x),  shape [B, D_z]

        Notes:
            When S > 1, you can broadcast x to [B, S, ...] during loss computation
            to evaluate log p(x | z_s) for each sample without copying x.
            For Bernoulli likelihoods, the decoder should output logits.
        """
        
        enc = self._encode(x, S=S) # VAEEncodeOutput(z, mu, log_var)
        dec = self._decode(enc.z)  # VAEDecodeOutput(x_hat)
        return VAEOutput(x_hat=dec.x_hat, z=enc.z, mu=enc.mu, log_var=enc.log_var)
    
    @torch.no_grad()
    def build(self, input_sample: torch.Tensor) -> None:
        f = self.encoder(input_sample)
        self.sampling_layer.build(f)
        assert self.sampling_layer.built, 'Sampling layer building failed.'
        self._built = True