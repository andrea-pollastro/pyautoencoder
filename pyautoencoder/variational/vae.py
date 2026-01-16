import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Union, Tuple

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

class AdaGVAE(VAE):
    r"""Adaptive Group Variational Autoencoder (Ada-GVAE), from Locatello et al. (2020).

    This class extends the VAE class and enables feature disentanglement in the latent space. 
    For inference, use the .encode() and .decode() methods, as the forward method expects pairs of images, 
    following the formulation introduced by Locatello et al.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        latent_dim: int,
    ):
        """Construct an AdaGVAE from an encoder, decoder, and latent size.

        Notes
        -----
        The encoder and decoder are identical to those in a standard VAE. The adaptive
        grouping mechanism is applied during the encoding step when processing paired inputs.

        Parameters
        ----------
        encoder : nn.Module
            Maps input ``x`` to a feature vector ``f(x)`` with shape ``[B, F]``.
        decoder : nn.Module
            Maps latent samples ``z`` to reconstructions ``x_hat``.
        latent_dim : int
            Dimensionality ``D_z`` of the latent space.
        """
        super().__init__(encoder=encoder, decoder=decoder, latent_dim=latent_dim)

    # --- training-time hooks required by BaseAutoencoder ---
    def _encode_pair(self, x1: torch.Tensor, x2: torch.Tensor, S: int = 1) -> Tuple[VAEEncodeOutput, VAEEncodeOutput]:
        r"""Encode a pair of inputs with adaptive posterior alignment.

        As described in Locatello et al., this method:

        1. Encodes both inputs independently to obtain posterior parameters ``(mu1, log_var1)`` and ``(mu2, log_var2)``.
        2. Computes element-wise KL divergence between the two posteriors: ``KL(q1||q2) → [B, D_z]``.
        3. Computes a per-sample threshold ``tau`` based on KL divergences.
        4. For each dimension, selects aligned (shared) or independent posteriors:
           - If ``KL(q1_d||q2_d) < tau``: uses average distribution ``q_tilde``.
           - If ``KL(q1_d||q2_d) ≥ tau``: uses original independent distribution.
        5. Samples from the resulting (mixed) posteriors.

        Parameters
        ----------
        x1 : torch.Tensor
            First input batch of shape ``[B, ...]``.
        x2 : torch.Tensor
            Second input batch of shape ``[B, ...]``.
        S : int
            Number of latent samples per input.

        Returns
        -------
        Tuple[VAEEncodeOutput, VAEEncodeOutput]
            A pair of ``VAEEncodeOutput`` objects, each containing:

            - ``z`` of shape ``[B, S, D_z]``: samples from the adapted posteriors.
            - ``mu`` of shape ``[B, D_z]``: the (adapted) means.
            - ``log_var`` of shape ``[B, D_z]``: the (adapted) log-variances.

        Notes
        -----
        The thresholding mechanism promotes learning of shared latent factors
        while allowing independent variation for high-divergence dimensions. 
        This encourages disentanglement and structured representations.
        """
        _, mu1, log_var1 = self.sampling_layer(self.encoder(x1))
        _, mu2, log_var2 = self.sampling_layer(self.encoder(x2))

        # KL(q1||q2) -> [B, latents]
        kl_q1_q2 = kl_divergence_diag_gaussian(mu1, log_var1, mu2, log_var2, reduce_sum=False)

        # Computing threshold tau
        max_delta = torch.max(kl_q1_q2, dim=1, keepdim=True)[0]
        min_delta = torch.min(kl_q1_q2, dim=1, keepdim=True)[0]
        tau = 0.5 * (max_delta + min_delta)

        # Computing q_tilde1 and q_tilde2
        mu_mean = 0.5*(mu1 + mu2)
        var_mean = 0.5*(torch.exp(log_var1) + torch.exp(log_var2))
        log_var_mean = torch.log(var_mean)

        mask = kl_q1_q2 < tau
        mu_tilde1 = torch.where(mask, mu_mean, mu1)
        mu_tilde2 = torch.where(mask, mu_mean, mu2)
        log_var_tilde1 = torch.where(mask, log_var_mean, log_var1)
        log_var_tilde2 = torch.where(mask, log_var_mean, log_var2)

        z1 = self.sampling_layer.reparametrize(mu=mu_tilde1, log_var=log_var_tilde1, S=S)
        z2 = self.sampling_layer.reparametrize(mu=mu_tilde2, log_var=log_var_tilde2, S=S)

        return VAEEncodeOutput(z=z1, mu=mu_tilde1, log_var=log_var_tilde1), \
               VAEEncodeOutput(z=z2, mu=mu_tilde2, log_var=log_var_tilde2)


    def forward(self, x1: torch.Tensor, x2: torch.Tensor, S: int = 1) -> Tuple[VAEOutput, VAEOutput]:
        """Full AdaGVAE forward pass: encode pairs with adaptive grouping, sample, and decode.

        Parameters
        ----------
        x1 : torch.Tensor
            First input batch of shape ``[B, ...]``.
        x2 : torch.Tensor
            Second input batch of shape ``[B, ...]``.
        S : int
            Number of latent samples for Monte Carlo estimates.

        Returns
        -------
        Tuple[VAEOutput, VAEOutput]
            A pair of VAE outputs, each containing:

            - ``x_hat``: reconstructions from the adapted latent samples.
            - ``z``: latent samples from the adapted posteriors.
            - ``mu``: (adapted) posterior means.
            - ``log_var``: (adapted) posterior log-variances.
        """
        x1_enc, x2_enc = self._encode_pair(x1, x2, S=S)
        x1_dec = self._decode(x1_enc.z)
        x2_dec = self._decode(x2_enc.z)
        return VAEOutput(x_hat=x1_dec.x_hat, z=x1_enc.z, mu=x1_enc.mu, log_var=x1_enc.log_var), \
               VAEOutput(x_hat=x2_dec.x_hat, z=x2_enc.z, mu=x2_enc.mu, log_var=x2_enc.log_var)
    
    def compute_loss(self,
                     x1: torch.Tensor, 
                     x1_vae_output: VAEOutput,
                     x2: torch.Tensor, 
                     x2_vae_output: VAEOutput,
                     beta: float = 1,
                     likelihood: Union[str, LikelihoodType] = LikelihoodType.GAUSSIAN) -> LossResult:
        r"""Compute the combined ELBO for a pair of inputs with adaptive posteriors.

        This method computes the sum of the standard VAE ELBOs for both inputs:

        .. math::

            \mathcal{L}(x_1, x_2; \beta)
                = \left[ \mathbb{E}_{q(\hat{z} \mid x_1)}[\log p(x_1 \mid \hat{z})]
                \;-\; \beta \, \mathrm{KL}(q(\hat{z} \mid x_1) \,\|\, p(\hat{z})) \right]
                + \left[ \mathbb{E}_{q(\hat{z} \mid x_2)}[\log p(x_2 \mid \hat{z})]
                \;-\; \beta \, \mathrm{KL}(q(\hat{z} \mid x_2) \,\|\, p(\hat{z})) \right].

        The key difference from standard VAE is that the posteriors :math:`q(\hat{z} | x_1)` and
        :math:`q(\hat{z} | x_2)` are obtained from the adaptive grouping mechanism, which can
        share dimensions based on KL divergence thresholds.

        Parameters
        ----------
        x1 : torch.Tensor
            First input batch of shape ``[B, ...]``.
        x1_vae_output : VAEOutput
            Output from the forward pass for ``x1``. Expected fields:

            - ``x_hat`` (torch.Tensor): Reconstructions, shape ``[B, ...]`` or ``[B, S, ...]``.
            - ``mu`` (torch.Tensor): (Adapted) posterior mean, shape ``[B, D_z]``.
            - ``log_var`` (torch.Tensor): (Adapted) posterior log-variance, shape ``[B, D_z]``.

        x2 : torch.Tensor
            Second input batch of shape ``[B, ...]``.
        x2_vae_output : VAEOutput
            Output from the forward pass for ``x2``. Same structure as ``x1_vae_output``.
        likelihood : Union[str, LikelihoodType], optional
            Likelihood model for the reconstruction term.
            Can be 'gaussian' or 'bernoulli'. Defaults to Gaussian.
        beta : float, optional
            Weighting factor for the KL term (beta-VAE).
            ``beta = 1`` yields the standard objective. Defaults to 1.

        Returns
        -------
        LossResult
            Result containing:

            * **objective** – Sum of negative ELBOs for both inputs (scalar).
            * **diagnostics** – Dictionary with:

              - ``"elbo"``: Sum of ELBOs for both inputs.
              - ``"log_likelihood_x1"``: Mean reconstruction term for ``x1``.
              - ``"log_likelihood_x2"``: Mean reconstruction term for ``x2``.
              - ``"kl_divergence_x1"``: Mean KL divergence for ``x1``'s posterior.
              - ``"kl_divergence_x2"``: Mean KL divergence for ``x2``'s posterior.

        Notes
        -----
        - All diagnostics are **batch means** (per-sample losses averaged over ``B``).
        - Gradients flow through both decoders; neither input is detached.
        - The adaptive grouping introduces implicit structure learning through
          the selective sharing of posterior dimensions.
        """
        x1_loss_info = super().compute_loss(x=x1, vae_output=x1_vae_output, beta=beta, likelihood=likelihood)
        x2_loss_info = super().compute_loss(x=x2, vae_output=x2_vae_output, beta=beta, likelihood=likelihood)

        return LossResult(
            objective = x1_loss_info.objective + x2_loss_info.objective,
            diagnostics = {
                'elbo': x1_loss_info.diagnostics['elbo'] + x2_loss_info.diagnostics['elbo'],
                'log_likelihood_x1': x1_loss_info.diagnostics['log_likelihood'],
                'log_likelihood_x2': x2_loss_info.diagnostics['log_likelihood'],
                'kl_divergence_x1': x1_loss_info.diagnostics['kl_divergence'],
                'kl_divergence_x2': x2_loss_info.diagnostics['kl_divergence'],
            }
        )