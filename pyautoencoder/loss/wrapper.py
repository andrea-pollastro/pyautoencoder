from dataclasses import dataclass
from typing import Dict, Optional, Union
import math
import torch

from .base import LikelihoodType, log_likelihood
from .vae import compute_ELBO
from ..vanilla.autoencoder import AEOutput
from ..variational.vae import VAEOutput

LN2 = math.log(2.0)
LOG_2PI = math.log(2.0 * math.pi)  # for Gaussian sigma^2=1 diagnostics

@dataclass
class LossComponents:
    """Container for a scalar loss, its components, and extra metrics.

    Attributes
    ----------
    total : torch.Tensor
        Scalar loss to optimize (already reduced over the batch).
    components : Dict[str, torch.Tensor]
        Dictionary of named scalar terms that compose the loss
        (for example ``"negative_log_likelihood"``,
        ``"beta_kl_divergence"``).
    metrics : Optional[Dict[str, torch.Tensor]]
        Optional dictionary of additional scalar diagnostics
        (for example per-dimension NLL in nats/bits, KL per latent
        dimension, or mean-squared error). These values are intended for
        logging and monitoring and do not affect optimization directly.

    Notes
    -----
    All values are batch means unless stated otherwise.
    """

    total: torch.Tensor
    components: Dict[str, torch.Tensor]
    metrics: Optional[Dict[str, torch.Tensor]] = None

class BaseLoss:
    """Abstract base class for loss wrappers.

    Subclasses must implement :meth:`__call__` to compute a loss and return
    a :class:`LossComponents` object.
    """

    def __call__(self, *args, **kwargs) -> LossComponents:
        """Compute the loss and return its components.

        Returns
        -------
        LossComponents
            Container with the scalar loss, its components, and optional
            diagnostics.

        Raises
        ------
        NotImplementedError
            Always raised in the base class; must be overridden by subclasses.
        """

        raise NotImplementedError

class VAELoss(BaseLoss):
    r"""Loss wrapper for (beta-)Variational Autoencoders.

    The optimized objective is the **negative ELBO**,

    .. math::

        \mathcal{L}(x; \beta)
            = -\mathbb{E}_{q(z \mid x)}[\log p(x \mid z)]
            + \beta \, \mathrm{KL}(q(z \mid x) \,\|\, p(z)).

    The class also reports several size-normalized diagnostics such as 
    per-dimension NLL (in nats and bits) and KL per latent dimension.
    """

    def __init__(
        self,
        beta: float = 1.0,
        likelihood: Union[str, LikelihoodType] = LikelihoodType.GAUSSIAN,
    ):
        r"""Create a VAE loss with a given :math:`\beta` and likelihood model.

        The negative ELBO is used as the optimization objective. The reconstruction
        term is based on the negative log-likelihood (NLL) of the reconstructions:

        - Gaussian (:math:`\sigma^2 = 1`):
          per-dimension NLL is

        .. math::

            \tfrac{1}{2}\bigl( (x - x_{\hat{}})^2 + \log(2\pi) \bigr).

        - Bernoulli (logits):
          per-dimension NLL is given by
          :func:`torch.nn.functional.binary_cross_entropy_with_logits`.

        Parameters
        ----------
        beta : float, optional 
            Weighting factor for the KL term (:math:`\beta`-VAE). ``beta = 1``
            corresponds to the standard VAE objective.
        likelihood : Union[str, LikelihoodType], optional 
            Likelihood model for :math:`p(x \mid z)`. Either a string
            (``"gaussian"``, ``"bernoulli"``) or a :class:`LikelihoodType` enum
            value. For the Gaussian case, unit variance :math:`\sigma^2 = 1` is
            assumed; for Bernoulli, ``x_hat`` is interpreted as logits.
        """


        self.beta = beta
        if isinstance(likelihood, str):
            likelihood = LikelihoodType(likelihood.lower())
        self.likelihood = likelihood

    def __call__(self, x: torch.Tensor, model_output: VAEOutput) -> LossComponents:
        r"""Compute VAE loss components and size-normalized diagnostics.

        This method wraps :func:`vae.compute_ELBO` and returns the negative ELBO
        as the optimization objective, together with additional metrics for
        monitoring.

        Parameters
        ----------
        x : torch.Tensor 
            Ground-truth inputs of shape ``[B, ...]``.
        model_output : VAEOutput
            Output from the VAE forward pass. Expected fields include:

            - ``x_hat`` (torch.Tensor): Reconstructed samples of shape
              ``[B, S, ...]``, where ``S`` is the number of Monte Carlo samples
              from :math:`q(z \mid x)`.
            - ``z`` (torch.Tensor): Latent samples (unused by this method).
            - ``mu`` (torch.Tensor): Mean of :math:`q(z \mid x)`, shape
              ``[B, D_z]``.
            - ``log_var`` (torch.Tensor): Log-variance of :math:`q(z \mid x)`,
              shape ``[B, D_z]``.

        Returns
        -------
        LossComponents
            ``LossComponents`` container with:

            * **total** – Scalar negative ELBO (to minimize).
            * **components** – Dictionary with:

              - ``"negative_log_likelihood"``:
                  batch-mean :math:`-\mathbb{E}_{q}[\log p(x \mid z)]` in nats.
              - ``"beta_kl_divergence"``:
                  batch-mean :math:`\beta\,\mathrm{KL}(q \,\|\, p)` in nats.

            * **metrics** – Dictionary with additional diagnostics:

              - ``"elbo"``: batch-mean ELBO in nats.
              - ``"nll_per_dim_nats"``:
                  :math:`-\mathbb{E}_{q}[\log p(x \mid z)] / D_x`
                  (nats per input dimension).
              - ``"nll_per_dim_bits"``:
                  bits per dimension
                  (``nll_per_dim_nats / ln(2)``).
              - ``"beta_kl_per_latent_dim_nats"``:
                  :math:`\beta\,\mathrm{KL}(q \,\|\, p) / D_z`
                  (nats per latent dimension).
              - ``"beta_kl_per_latent_dim_bits"``:
                  bits per latent dimension
                  (``beta_kl_per_latent_dim_nats / ln(2)``).
              - ``"mse_per_dim"`` (Gaussian only):
                  per-dimension MSE derived from the Gaussian identity.
              - ``"mse"`` (Gaussian only):
                  MSE derived from the Gaussian identity.

        Notes
        -----
        Reductions follow:

        1. Sum over feature dimensions.
        2. Average over Monte Carlo samples (if any).
        3. Average over the batch.

        For the Gaussian (:math:`\sigma^2=1`) case, the per-dimension MSE is
        computed as

        .. math::

            \text{MSE}_{\text{per dim}}
                = 2\,\text{NLL}_{\text{per dim}} - \log(2\pi),

        and clamped to be non-negative.
        """

        x_hat = model_output.x_hat
        mu = model_output.mu
        log_var = model_output.log_var

        elbo_components = compute_ELBO(
            x=x,
            x_hat=x_hat,
            mu=mu,
            log_var=log_var,
            likelihood=self.likelihood,
            beta=self.beta,
        )

        D_x = x[0].numel()
        D_z = mu.size(-1)

        # Per-dimension / per-latent-dimension metrics
        nll_per_dim_nats = -elbo_components.log_likelihood / D_x          # nats/dim
        nll_per_dim_bits = nll_per_dim_nats / LN2                         # bits/dim

        beta_kl_per_latent_dim_nats = elbo_components.beta_kl_divergence / D_z      # nats/latent-dim
        beta_kl_per_latent_dim_bits = beta_kl_per_latent_dim_nats / LN2             # bits/latent-dim

        metrics: Dict[str, torch.Tensor] = {
            'elbo': elbo_components.elbo.detach().cpu(),
            'nll_per_dim_nats': nll_per_dim_nats.detach().cpu(),
            'nll_per_dim_bits': nll_per_dim_bits.detach().cpu(),
            'beta_kl_per_latent_dim_nats': beta_kl_per_latent_dim_nats.detach().cpu(),
            'beta_kl_per_latent_dim_bits': beta_kl_per_latent_dim_bits.detach().cpu(),
        }

        # Extra: derive MSE/dim for Gaussian(sigma^2=1)
        if self.likelihood == LikelihoodType.GAUSSIAN:
            # NLL_per_dim = 0.5*MSE_per_dim + 0.5*log(2pi) ⇒ MSE_per_dim = 2*NLL_per_dim - log(2pi)
            mse_per_dim = torch.clamp(2.0 * nll_per_dim_nats - LOG_2PI, min=0.0)
            metrics['mse_per_dim'] = mse_per_dim.detach().cpu()
            mse = mse_per_dim * D_x
            metrics['mse'] = mse.detach().cpu()

        return LossComponents(
            total=-elbo_components.elbo,  # minimize negative ELBO
            components={
                'negative_log_likelihood': -elbo_components.log_likelihood,
                'beta_kl_divergence': elbo_components.beta_kl_divergence,
            },
            metrics=metrics,
        )

class AELoss(BaseLoss):
    r"""Loss wrapper for standard Autoencoders.

    The loss is the reconstruction negative log-likelihood (NLL), computed
    under either a Gaussian (:math:`\sigma^2 = 1`) or Bernoulli likelihood.
    In addition to the scalar NLL, the class reports per-dimension metrics
    (in nats and bits) and, for the Gaussian case, a derived MSE per input
    dimension.
    """

    def __init__(self, likelihood: Union[str, LikelihoodType] = LikelihoodType.GAUSSIAN):
        r"""Create an Autoencoder loss with a chosen likelihood model.

        The reconstruction loss is based on the negative log-likelihood (NLL)
        of the reconstructions:

        - Gaussian (:math:`\sigma^2 = 1`):
          per-dimension NLL is

        .. math::

            \tfrac{1}{2}\bigl( (x - x_{\hat{}})^2 + \log(2\pi) \bigr).

        - Bernoulli (logits):
          per-dimension NLL is given by
          :func:`torch.nn.functional.binary_cross_entropy_with_logits`.

        Parameters
        ----------
        likelihood : Union[str, LikelihoodType], optional 
            Likelihood model for :math:`p(x \mid z)`. Either a string
            (``"gaussian"``, ``"bernoulli"``) or a :class:`LikelihoodType` enum
            value. For the Gaussian case, unit variance :math:`\sigma^2 = 1` is
            assumed; for Bernoulli, ``x_hat`` is interpreted as logits.
        """

        if isinstance(likelihood, str):
            likelihood = LikelihoodType(likelihood.lower())
        self.likelihood = likelihood

    def __call__(self, x: torch.Tensor, model_output: AEOutput) -> LossComponents:
        r"""Compute Autoencoder reconstruction loss and diagnostics.

        The scalar loss is the batch-mean reconstruction NLL. Additional metrics
        are provided to monitor model behaviour in a size-normalized way.

        Parameters
        ----------
        x : torch.Tensor
            Ground-truth inputs of shape ``[B, ...]``.
        model_output : AEOutput
            Output from the AE forward pass. Expected fields include:

            - ``x_hat`` (torch.Tensor): Reconstructions, shape ``[B, ...]``.
            - ``z`` (torch.Tensor): Latent representation (unused by this method).

        Returns
        -------
        LossComponents
            ``LossComponents`` with:

            * **total** – Scalar batch-mean reconstruction loss (NLL in nats).
            * **components** – Dictionary with:

              - ``"negative_log_likelihood"``:
                  same scalar as ``total``.

            * **metrics** – Dictionary with diagnostics:

              - ``"nll_per_dim_nats"``:
                  :math:`\text{NLL} / D_x` (nats per input dimension).
              - ``"nll_per_dim_bits"``:
                  bits per dimension (``nll_per_dim_nats / ln(2)``).
              - ``"mse_per_dim"`` (Gaussian only):
                  per-dimension MSE derived from the Gaussian identity.
              - ``"mse"`` (Gaussian only):
                  MSE derived from the Gaussian identity.

        Notes
        -----
        Reductions follow: 
        
        1. Elementwise log-likelihood 
        2. Sum over feature dimensions
        3. Mean over the batch.

        For the Gaussian (:math:`\sigma^2=1`) case, the per-dimension MSE is
        computed as

        .. math::

            \text{MSE}_{\text{per dim}}
                = 2\,\text{NLL}_{\text{per dim}} - \log(2\pi),

        and clamped to be non-negative.

        Ensure that inputs match the chosen likelihood:

        - Gaussian: continuous data (typically standardized).
        - Bernoulli: targets in :math:`[0, 1]`, predictions given as logits.
        """

        x_hat = model_output.x_hat

        B = x.size(0)
        D_x = x[0].numel()

        # Elementwise log-likelihood → per-sample sum → batch mean
        ll_elem = log_likelihood(x, x_hat, likelihood=self.likelihood)         # [B, ...]
        ll_per_sample = ll_elem.reshape(B, -1).sum(-1)                         # [B]
        nll = (-ll_per_sample).mean()                                          # scalar NLL

        # Per-dim diagnostics
        nll_per_dim_nats = nll / D_x                                           # nats/dim
        nll_per_dim_bits = nll_per_dim_nats / LN2                              # bits/dim

        metrics: Dict[str, torch.Tensor] = {
            'nll_per_dim_nats': nll_per_dim_nats.detach().cpu(),
            'nll_per_dim_bits': nll_per_dim_bits.detach().cpu(),
        }

        if self.likelihood == LikelihoodType.GAUSSIAN:
            mse_per_dim = torch.clamp(2.0 * nll_per_dim_nats - LOG_2PI, min=0.0)
            metrics['mse_per_dim'] = mse_per_dim.detach().cpu()
            mse = mse_per_dim * D_x
            metrics['mse'] = mse.detach().cpu()

        return LossComponents(
            total=nll,
            components={'negative_log_likelihood': nll},
            metrics=metrics,
        )
