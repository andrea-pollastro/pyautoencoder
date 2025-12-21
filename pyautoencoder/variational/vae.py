import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Union, Dict

from ..loss.base import (
    LikelihoodType, 
    log_likelihood, 
    kl_divergence_diag_gaussian, 
    LossResult
)
from .._base.base import BaseAutoencoder, ModelOutput
from .stochastic_layers import FullyFactorizedGaussian

@dataclass(slots=True, repr=False)
class VAEEncodeOutput(ModelOutput):
    r"""Output of the VAE encoder stage.

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
    r"""Output of the VAE decoder stage.

    Attributes
    ----------
    x_hat : torch.Tensor
        Reconstructions or logits of shape ``[B, S, ...]``.
    """

    x_hat: torch.Tensor

@dataclass(slots=True, repr=False)
class VAEOutput(ModelOutput):
    r"""Output of a full VAE forward pass.

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

        Notes
        -----
        A sampling layer is internally created using a fully factorized Gaussian
        (`FullyFactorizedGaussian`). At the moment this sampling layer is not
        configurable from the outside: it is fixed and not exposed as an argument
        to the constructor.

        In a future revision, the sampling layer will become a user-selectable
        component, allowing different reparameterization modules to be passed in.
        The VAE will then choose the appropriate sampling strategy based on a
        constructor parameter.

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
        r"""Encode inputs and draw Monte Carlo latent samples.

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

    def compute_loss(self,
                     x: torch.Tensor, 
                     vae_output: VAEOutput,
                     beta: float = 1,
                     likelihood: Union[str, LikelihoodType] = LikelihoodType.GAUSSIAN) -> LossResult:
        r"""Compute the Evidence Lower Bound (ELBO) for a (beta-)Variational Autoencoder.

        This method implements the beta-VAE objective:

        .. math::

            \mathcal{L}(x; \beta)
                = \mathbb{E}_{q(z \mid x)}[\log p(x \mid z)]
                \;-\;
                \beta \, \mathrm{KL}(q(z \mid x) \,\|\, p(z)).

        The reconstruction term :math:`\log p(x \mid z)` is computed using
        :func:`loss.base.log_likelihood`, which supports both Gaussian and
        Bernoulli likelihoods.

        Monte Carlo estimation
        ----------------------
        If ``x_hat`` in ``vae_output`` contains ``S`` Monte Carlo samples, 
        the expectation :math:`\mathbb{E}_{q(z \mid x)}` is approximated by:

        .. math::

            \mathbb{E}_{q(z \mid x)}[\log p(x \mid z)]
                \approx \frac{1}{S} \sum_{s=1}^{S}
                \log p(x \mid z^{(s)}).

        Broadcasting
        ----------------------
        - If ``x_hat`` has shape ``[B, ...]``, it is expanded to ``[B, 1, ...]``.
        - ``x`` is broadcast to match the sample dimension of ``x_hat``.

        Parameters
        ----------
        x : torch.Tensor
            Ground-truth inputs, shape ``[B, ...]``.
        vae_output : VAEOutput
            Output from the VAE forward pass. Expected fields include:

            - ``x_hat`` (torch.Tensor): Reconstructed samples, shape ``[B, ...]`` or ``[B, S, ...]``.
            - ``mu`` (torch.Tensor): Mean of :math:`q(z \mid x)`, shape ``[B, D_z]``.
            - ``log_var`` (torch.Tensor): Log-variance of :math:`q(z \mid x)`, shape ``[B, D_z]``.

        likelihood : Union[str, LikelihoodType], optional
            Likelihood model for the reconstruction term. 
            Can be 'gaussian' or 'bernoulli'. Defaults to Gaussian.
        beta : float, optional
            Weighting factor for the KL term (beta-VAE). 
            ``beta = 1`` yields the standard VAE. Defaults to 1.

        Returns
        -------
        LossResult
            Result containing:
            
            * **objective** – Negative mean ELBO (scalar).
            * **diagnostics** – Dictionary with:

              - ``"elbo"``: Mean ELBO over the batch.
              - ``"log_likelihood"``: Mean reconstruction term :math:`\mathbb{E}_{q}[\log p(x \mid z)]`.
              - ``"kl_divergence"``: Mean :math:`\mathrm{KL}(q \,\|\, p)` over the batch.

        Notes
        -----
        - All returned diagnostics are **batch means**.
        - Gradients flow through the decoder; neither input is detached.
        """
        x_hat = vae_output.x_hat
        mu = vae_output.mu
        log_var = vae_output.log_var

        if isinstance(likelihood, str):
            likelihood = LikelihoodType(likelihood.lower())

        # Ensure a sample dimension S exists -> [B, S, ...]
        if x_hat.dim() == x.dim():
            x_hat = x_hat.unsqueeze(1)  # S = 1
        B, S = x_hat.size(0), x_hat.size(1)

        # Broadcast x to match x_hat's [B, S, ...] shape
        x_expanded = x.unsqueeze(1)  # [B, 1, ...]
        if x_expanded.shape != x_hat.shape:
            # expand_as is a view (no real data copy) when only singleton dims are expanded
            x_expanded = x_expanded.expand_as(x_hat)

        # log p(x|z): elementwise -> sum over features => [B, S]
        log_px_z = log_likelihood(x_expanded, x_hat, likelihood=likelihood)
        log_px_z = log_px_z.reshape(B, S, -1).sum(-1)

        # E_q[log p(x|z)] via Monte Carlo average across S: [B]
        E_log_px_z = log_px_z.mean(dim=1)

        # KL(q||p): [B]
        kl_q_p = kl_divergence_diag_gaussian(mu, log_var)

        # ELBO per sample and batch means (retain grads)
        elbo_per_sample = E_log_px_z - beta * kl_q_p          # [B]
        elbo = elbo_per_sample.mean()                         # scalar

        return LossResult(
            objective = -elbo,
            diagnostics = {
                'elbo': elbo.item(),
                'log_likelihood': E_log_px_z.mean().item(),
                'kl_divergence': kl_q_p.mean().item(),
            }
        )
