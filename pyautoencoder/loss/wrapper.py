"""Loss wrappers for easy loss computation and tracking."""
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
    """
    Container for loss components with detailed metrics.

    Args:
        total (torch.Tensor): Scalar loss to optimize (already reduced over batch).
        components (Dict[str, torch.Tensor]): Named scalar terms that compose the loss
            (e.g., 'negative_log_likelihood', 'beta_kl_divergence').
        metrics (Optional[Dict[str, torch.Tensor]]): Additional scalar diagnostics
            (e.g., per-dimension metrics in nats/bits, KL per latent dimension).

    Notes:
        - All values are batch means unless specified otherwise.
        - Metrics are intended for logging/monitoring and do not affect optimization directly.
    """
    total: torch.Tensor
    components: Dict[str, torch.Tensor]
    metrics: Optional[Dict[str, torch.Tensor]] = None

class BaseLoss:
    """Base class for all loss functions."""
    def __call__(self, *args, **kwargs) -> LossComponents:
        raise NotImplementedError

class VAELoss(BaseLoss):
    def __init__(
        self,
        beta: float = 1.0,
        likelihood: Union[str, LikelihoodType] = LikelihoodType.GAUSSIAN,
    ):
        """
        Loss function for Variational Autoencoders (beta-VAE style).
        The optimized loss is the *negative ELBO*. 
        Uses negative log-likelihood (NLL) of reconstructions:
        - Gaussian (sigma^2=1): per-dim NLL = 0.5*[(x - x_hat)^2 + log(2pi)].
        - Bernoulli (logits): per-dim NLL = BCEWithLogits(x_hat, x).

        Args:
            beta (float): Weighting factor for the KL term (beta-VAE).
            likelihood (Union[str, LikelihoodType]): Likelihood model for p(x|z).
                Supported: 'gaussian' (sigma^2=1) or 'bernoulli' (logits).
        """
        self.beta = beta
        self.likelihood = likelihood

    def __call__(self, x: torch.Tensor, model_output: VAEOutput) -> LossComponents:
        """
        Computes VAE loss components and size-normalized diagnostics.

        Args:
            x (torch.Tensor): Ground truth inputs, shape [B, ...].
            model_output (VAEOutput): from the VAE forward pass, dataclass with:
                - x_hat (torch.Tensor): Reconstructed samples, shape [B, S, ...],
                                        where S is the number of Monte Carlo samples from q(z|x).
                - z (torch.Tensor): Latent samples (unused here).
                - mu (torch.Tensor): Mean of q(z|x), shape [B, D_z].
                - log_var (torch.Tensor): Log-variance of q(z|x), shape [B, D_z].

        Returns:
            LossComponents: Named container with:
                - total (torch.Tensor): Scalar, negative ELBO (to minimize).
                - components (Dict[str, torch.Tensor]):
                    * 'negative_log_likelihood': Scalar, batch-mean -E_q[log p(x|z)] (nats).
                    * 'beta_kl_divergence': Scalar, batch-mean beta * KL(q||p) (nats).
                - metrics (Dict[str, torch.Tensor]):
                    * 'elbo': Scalar, batch-mean ELBO (nats).
                    * 'nll_per_dim_nats': Scalar, -E_q[log p(x|z)] / D_x (nats/dim).
                    * 'nll_per_dim_bits': Scalar, bits per dimension = nll_per_dim_nats / ln(2) (bits/dim).
                    * 'beta_kl_per_latent_dim_nats': Scalar, beta * KL / D_z (nats per latent dim).
                    * 'beta_kl_per_latent_dim_bits': Scalar, beta_kl_per_latent_dim_nats / ln(2) (bits per latent dim).
                    * 'mse_per_dim' (optional): Scalar, derived from Gaussian sigma^2=1 identity.

        Notes:
            - Reductions follow: sum over feature dimensions → mean over MC samples (if any) → mean over batch.
            - log p(x|z) is computed by `log_likelihood`:
                * Gaussian (sigma^2=1): per-dim NLL = 0.5*MSE + 0.5*log(2pi).
                * Bernoulli (logits): per-dim NLL = BCEWithLogits.
            - For Gaussian (sigma^2=1), 'mse_per_dim' is computed via:
                MSE_per_dim = 2*NLL_per_dim - log(2pi), clamped to be >= 0.
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

        return LossComponents(
            total=-elbo_components.elbo,  # minimize negative ELBO
            components={
                'negative_log_likelihood': -elbo_components.log_likelihood,
                'beta_kl_divergence': elbo_components.beta_kl_divergence,
            },
            metrics=metrics,
        )

class AELoss(BaseLoss):
    def __init__(self, likelihood: Union[str, LikelihoodType] = LikelihoodType.GAUSSIAN):
        """
        Loss function for standard Autoencoders.

        Uses negative log-likelihood (NLL) of reconstructions:
            - Gaussian (sigma^2=1): per-dim NLL = 0.5*[(x - x_hat)^2 + log(2pi)].
            - Bernoulli (logits): per-dim NLL = BCEWithLogits(x_hat, x).

        Args:
            likelihood (Union[str, LikelihoodType]): Likelihood model for p(x|z).
                Supported: 'gaussian' (sigma^2=1) or 'bernoulli' (logits).
        """
        self.likelihood = likelihood

    def __call__(self, x: torch.Tensor, model_output: AEOutput) -> LossComponents:
        """
        Computes Autoencoder reconstruction loss and size-normalized diagnostics.

        Args:
            x (torch.Tensor): Ground truth inputs, shape [B, ...].
            model_output (AEOutput): from the AE forward pass, dataclass containing:
                - x_hat (torch.Tensor): Reconstructions, shape [B, ...].
                - z (torch.Tensor): Latent samples (unused here).

        Returns:
            LossComponents: Named container with:
                - total (torch.Tensor): Scalar, batch-mean reconstruction loss (NLL in nats).
                - components (Dict[str, torch.Tensor]):
                    * 'negative_log_likelihood': Scalar, same as total.
                - metrics (Dict[str, torch.Tensor]):
                    * 'nll_per_dim_nats': Scalar, NLL / D_x (nats/dim).
                    * 'nll_per_dim_bits': Scalar, bits per dimension = nll_per_dim_nats / ln(2) (bits/dim).
                    * 'mse_per_dim' (optional): Scalar, derived for Gaussian sigma^2=1.

        Notes:
            - Reductions follow: elementwise log-likelihood → sum over feature dimensions
              → mean over batch.
            - For Gaussian (sigma^2=1), 'mse_per_dim' is computed via:
                MSE_per_dim = 2*(NLL_per_dim) - log(2pi), clamped to be >= 0.
            - Ensure inputs match the likelihood’s expected scale:
                * Gaussian: continuous data (typically standardized).
                * Bernoulli: targets in [0, 1], predictions given as logits.
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

        return LossComponents(
            total=nll,
            components={'negative_log_likelihood': nll},
            metrics=metrics,
        )
