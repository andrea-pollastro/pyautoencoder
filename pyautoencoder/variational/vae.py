import torch
import torch.nn as nn
from dataclasses import dataclass
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
        Latent samples of shape ``[B, S, D_z]``, produced by
        :meth:`VAE._encode` or :meth:`VAE.encode`.
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
        Reconstructions or logits of shape ``[B, S, ...]``, produced by
        :meth:`VAE._decode` or :meth:`VAE.decode`.
    """

    x_hat: torch.Tensor

@dataclass(slots=True, repr=False)
class VAEOutput(ModelOutput):
    r"""Output of a full VAE forward pass.

    Attributes
    ----------
    x_hat : torch.Tensor
        Reconstructions or logits of shape ``[B, S, ...]``, produced by
        :meth:`VAE.forward`.
    z : torch.Tensor
        Latent samples of shape ``[B, S, D_z]``, produced by
        :meth:`VAE.forward`.
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

        Notes
        -----
        A :class:`FullyFactorizedGaussian` sampling layer is created internally
        and not exposed as a constructor parameter.
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
        S : int, optional
            Number of latent samples per input. Defaults to 1.

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
        S : int, optional
            Number of latent samples for Monte Carlo estimates. Defaults to 1.

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
        
        enc = self._encode(x, S=S)
        dec = self._decode(enc.z)
        return VAEOutput(x_hat=dec.x_hat, z=enc.z, mu=enc.mu, log_var=enc.log_var)
    
    def compute_loss(self,
                     x: torch.Tensor, 
                     vae_output: VAEOutput,
                     beta: float = 1,
                     likelihood: str | LikelihoodType = LikelihoodType.GAUSSIAN) -> LossResult:
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
        ------------
        - If ``x_hat`` has shape ``[B, ...]``, it is expanded to ``[B, 1, ...]``.
        - ``x`` is broadcast to match the sample dimension of ``x_hat``.

        Parameters
        ----------
        x : torch.Tensor
            Ground-truth inputs of shape ``[B, ...]``.
        vae_output : VAEOutput
            Output from the VAE forward pass. Expected fields include:

            - ``x_hat`` (torch.Tensor): Reconstructed samples, shape ``[B, ...]`` or ``[B, S, ...]``.
            - ``mu`` (torch.Tensor): Mean of :math:`q(z \mid x)`, shape ``[B, D_z]``.
            - ``log_var`` (torch.Tensor): Log-variance of :math:`q(z \mid x)`, shape ``[B, D_z]``.

        beta : float, optional
            Weighting factor for the KL term (beta-VAE).
            ``beta = 1`` yields the standard VAE. Defaults to 1.
        likelihood : str | LikelihoodType, optional
            Likelihood model for the reconstruction term
            (``'gaussian'`` or ``'bernoulli'``). Defaults to Gaussian.

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

@dataclass(slots=True, repr=False)
class AdaGVAEOutput(ModelOutput):
    r"""Output of a full AdaGVAE paired forward pass.

    Attributes
    ----------
    output1 : VAEOutput
        VAE output for the first input of the pair, with adapted posterior.
    output2 : VAEOutput
        VAE output for the second input of the pair, with adapted posterior.
    """

    output1: VAEOutput
    output2: VAEOutput

class AdaGVAE(nn.Module):
    """Adaptive Group Variational Autoencoder (Ada-GVAE), from Locatello et al. (2020).

    Wraps a :class:`VAE` and adds adaptive posterior grouping for feature disentanglement.
    All VAE parameters are tracked through this wrapper.

    :meth:`forward` expects a pair of inputs ``(x1, x2)`` and returns an
    :class:`AdaGVAEOutput` with adapted latent representations for both.
    For single-image inference after training, use ``model.vae.encode`` and
    ``model.vae.decode`` directly.
    """

    def __init__(self, vae: VAE):
        """Wrap a VAE with adaptive posterior grouping.

        Parameters
        ----------
        vae : VAE
            A configured :class:`VAE` instance whose encoder, decoder, and sampling
            layer are reused for the paired training objective.
        """
        super().__init__()
        self.vae = vae

    def build(self, input_sample: torch.Tensor) -> None:
        """Materialize lazy layers by delegating to the wrapped VAE's build step.

        Parameters
        ----------
        input_sample : torch.Tensor
            A representative input batch passed to :meth:`VAE.build`.
            Only the shape matters; values are not used.
        """
        self.vae.build(input_sample)

    def _encode_pair(self, x1: torch.Tensor, x2: torch.Tensor, S: int = 1) -> tuple[VAEEncodeOutput, VAEEncodeOutput]:
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
        S : int, optional
            Number of latent samples per input. Defaults to 1.

        Returns
        -------
        tuple[VAEEncodeOutput, VAEEncodeOutput]
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
        mu1, log_var1 = self.vae.sampling_layer.get_params(self.vae.encoder(x1))
        mu2, log_var2 = self.vae.sampling_layer.get_params(self.vae.encoder(x2))

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

        z1 = self.vae.sampling_layer._reparametrize(mu=mu_tilde1, log_var=log_var_tilde1, S=S)
        z2 = self.vae.sampling_layer._reparametrize(mu=mu_tilde2, log_var=log_var_tilde2, S=S)

        return (
            VAEEncodeOutput(z=z1, mu=mu_tilde1, log_var=log_var_tilde1),
            VAEEncodeOutput(z=z2, mu=mu_tilde2, log_var=log_var_tilde2),
        )

    def forward(self, x: tuple[torch.Tensor, torch.Tensor], S: int = 1) -> AdaGVAEOutput:
        """AdaGVAE training pass on a pair of images.

        For single-image inference after training use ``model.vae.encode``
        and ``model.vae.decode``.

        Parameters
        ----------
        x : tuple[torch.Tensor, torch.Tensor]
            A ``(x1, x2)`` pair, each of shape ``[B, ...]``.
        S : int, optional
            Number of latent samples for Monte Carlo estimates. Defaults to 1.

        Returns
        -------
        AdaGVAEOutput
            Adapted pair outputs containing reconstructions and posterior parameters
            for both inputs.
        """
        x1, x2 = x
        x1_enc, x2_enc = self._encode_pair(x1, x2, S=S)
        x1_dec = self.vae._decode(x1_enc.z)
        x2_dec = self.vae._decode(x2_enc.z)
        return AdaGVAEOutput(
            output1=VAEOutput(x_hat=x1_dec.x_hat, z=x1_enc.z, mu=x1_enc.mu, log_var=x1_enc.log_var),
            output2=VAEOutput(x_hat=x2_dec.x_hat, z=x2_enc.z, mu=x2_enc.mu, log_var=x2_enc.log_var),
        )

    def compute_loss(self,
                     x: tuple[torch.Tensor, torch.Tensor],
                     vae_output: AdaGVAEOutput,
                     beta: float = 1,
                     likelihood: str | LikelihoodType = LikelihoodType.GAUSSIAN) -> LossResult:
        r"""Compute the combined ELBO for a pair of inputs with adapted posteriors.

        .. math::

            \mathcal{L}(x_1, x_2; \beta)
                = \left[ \mathbb{E}_{q(\hat{z} \mid x_1)}[\log p(x_1 \mid \hat{z})]
                \;-\; \beta \, \mathrm{KL}(q(\hat{z} \mid x_1) \,\|\, p(\hat{z})) \right]
                + \left[ \mathbb{E}_{q(\hat{z} \mid x_2)}[\log p(x_2 \mid \hat{z})]
                \;-\; \beta \, \mathrm{KL}(q(\hat{z} \mid x_2) \,\|\, p(\hat{z})) \right].

        Parameters
        ----------
        x : tuple[torch.Tensor, torch.Tensor]
            The ``(x1, x2)`` pair of ground-truth inputs, each of shape ``[B, ...]``.
        vae_output : AdaGVAEOutput
            Output from :meth:`forward` called in training mode.
        beta : float, optional
            KL weighting factor. ``beta = 1`` yields the standard objective.
            Defaults to 1.
        likelihood : str | LikelihoodType, optional
            Likelihood model for the reconstruction term (``'gaussian'`` or
            ``'bernoulli'``). Defaults to Gaussian.

        Returns
        -------
        LossResult
            Result containing:

            * **objective** – Sum of negative ELBOs for both inputs (scalar).
            * **diagnostics** – Dictionary with:

              - ``"elbo"``: Sum of mean ELBOs for both inputs.
              - ``"log_likelihood_x1"``: Mean reconstruction term for ``x1``.
              - ``"log_likelihood_x2"``: Mean reconstruction term for ``x2``.
              - ``"kl_divergence_x1"``: Mean KL divergence for ``x1``.
              - ``"kl_divergence_x2"``: Mean KL divergence for ``x2``.
        """
        x1, x2 = x
        loss1 = self.vae.compute_loss(x=x1, vae_output=vae_output.output1, beta=beta, likelihood=likelihood)
        loss2 = self.vae.compute_loss(x=x2, vae_output=vae_output.output2, beta=beta, likelihood=likelihood)
        return LossResult(
            objective=loss1.objective + loss2.objective,
            diagnostics={
                'elbo': loss1.diagnostics['elbo'] + loss2.diagnostics['elbo'],
                'log_likelihood_x1': loss1.diagnostics['log_likelihood'],
                'log_likelihood_x2': loss2.diagnostics['log_likelihood'],
                'kl_divergence_x1': loss1.diagnostics['kl_divergence'],
                'kl_divergence_x2': loss2.diagnostics['kl_divergence'],
            }
        )