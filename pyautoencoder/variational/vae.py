import torch
import torch.nn as nn
from dataclasses import dataclass

from .._base.base import BaseAutoencoder, ModelOutput
from .stochastic_layers import FullyFactorizedGaussian

@dataclass(slots=True, repr=False)
class VAEEncodeOutput(ModelOutput):
    """Output of the VAE encoder stage.

    Attributes
    ----------
    z : torch.Tensor
        Latent samples of shape ``[B, S, D_z]`` (with ``S = 1`` allowed).
    mu : torch.Tensor
        Mean of the approximate posterior ``q(z \mid x)``, shape ``[B, D_z]``.
    log_var : torch.Tensor
        Log-variance of ``q(z \mid x)``, shape ``[B, D_z]``.
    """

    z: torch.Tensor
    mu: torch.Tensor
    log_var: torch.Tensor

@dataclass(slots=True, repr=False)
class VAEDecodeOutput(ModelOutput):
    """Output of the VAE decoder stage.

    Attributes
    ----------
    x_hat : torch.Tensor
        Reconstructions or logits of shape ``[B, S, ...]``.
    """

    x_hat: torch.Tensor


@dataclass(slots=True, repr=False)
class VAEOutput(ModelOutput):
    """Output of a full VAE forward pass.

    Attributes
    ----------
    x_hat : torch.Tensor
        Reconstructions or logits, shape ``[B, S, ...]``.
    z : torch.Tensor
        Latent samples, shape ``[B, S, D_z]``.
    mu : torch.Tensor
        Mean of ``q(z \mid x)``, shape ``[B, D_z]``.
    log_var : torch.Tensor
        Log-variance of ``q(z \mid x)``, shape ``[B, D_z]``.
    """

    x_hat: torch.Tensor
    z: torch.Tensor
    mu: torch.Tensor
    log_var: torch.Tensor

class VAE(BaseAutoencoder):
    r"""Variational Autoencoder following Kingma & Welling (2013).

    The model consists of:

    * an encoder mapping ``x → f(x)`` (feature representation),
    * a fully factorized Gaussian head producing ``(z, mu, log_var)``,
    * a decoder mapping latent samples ``z → x_hat``.

    Training uses Monte Carlo samples ``z`` for the reparameterization trick;
    evaluation mode returns deterministic repeated means.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        latent_dim: int,
    ):
        """Construct a Variational Autoencoder from an encoder, decoder, and latent size.

        Parameters
        ----------
        encoder : nn.Module
            Maps input ``x`` to a feature vector ``f(x)`` with shape ``[B, F]``.
        decoder : nn.Module
            Maps latent samples ``z`` to reconstructions ``x_hat``.
        latent_dim : int
            Dimensionality ``D_z`` of the latent space.
        """

        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.sampling_layer = FullyFactorizedGaussian(latent_dim=latent_dim)

    # --- training-time hooks required by BaseAutoencoder ---
    def _encode(self, x: torch.Tensor, S: int = 1) -> VAEEncodeOutput:
        """Encode inputs and draw Monte Carlo latent samples.

        Parameters
        ----------
        x : torch.Tensor
            Input batch of shape ``[B, ...]``. The encoder must output a flat
            feature vector per sample suitable for the sampling layer.
        S : int
            Number of latent samples per input.

        Returns
        -------
        VAEEncodeOutput
            Contains ``z`` of shape ``[B, S, D_z]``, and ``mu`` and ``log_var`` of
            shape ``[B, D_z]``.

        Notes
        -----
        The sampling layer behaves as:

        * ``train()`` – sample from ``q(z \mid x)``.
        * ``eval()`` – return tiled means for deterministic evaluation.
        """

        f = self.encoder(x)
        z, mu, log_var = self.sampling_layer(f, S=S)
        return VAEEncodeOutput(z=z, mu=mu, log_var=log_var)

    def _decode(self, z: torch.Tensor) -> VAEDecodeOutput:
        """Decode latent variables into reconstructions.

        Parameters
        ----------
        z : torch.Tensor
            Latent samples of shape ``[B, S, D_z]``.

        Returns
        -------
        VAEDecodeOutput
            Contains ``x_hat`` of shape ``[B, S, ...]``.
        """

        B, S, D_z = z.shape
        x_hat_flat = self.decoder(z.reshape(B * S, D_z))  # [B * S, ...]
        x_hat = x_hat_flat.reshape(B, S, *x_hat_flat.shape[1:])
        return VAEDecodeOutput(x_hat=x_hat)

    def forward(self, x: torch.Tensor, S: int = 1) -> VAEOutput:
        """Full VAE pass: encode, sample ``S`` times, decode.

        Parameters
        ----------
        x : torch.Tensor
            Input batch of shape ``[B, ...]``.
        S : int
            Number of latent samples for Monte Carlo estimates.

        Returns
        -------
        VAEOutput
            Contains reconstructions ``x_hat``, latent samples ``z``, and the
            posterior parameters ``mu`` and ``log_var``.

        Notes
        -----
        If ``S > 1``, loss computation can broadcast ``x`` to shape
        ``[B, S, ...]`` without materializing copies. For Bernoulli likelihoods,
        the decoder must output logits.
        """
        
        enc = self._encode(x, S=S) # VAEEncodeOutput(z, mu, log_var)
        dec = self._decode(enc.z)  # VAEDecodeOutput(x_hat)
        return VAEOutput(x_hat=dec.x_hat, z=enc.z, mu=enc.mu, log_var=enc.log_var)
    
    @torch.no_grad()
    def build(self, input_sample: torch.Tensor) -> None:
        """Build the VAE using a representative input sample.

        The encoder is applied to ``input_sample`` to obtain feature vectors,
        which are then used to build the Gaussian sampling layer. Once the
        sampling layer is built, the VAE is marked as constructed.

        Parameters
        ----------
        input_sample : torch.Tensor
            Example input tensor used to infer encoder feature dimensionality.
        """

        f = self.encoder(input_sample)
        self.sampling_layer.build(f)
        assert self.sampling_layer.built, 'Sampling layer building failed.'
        self._built = True