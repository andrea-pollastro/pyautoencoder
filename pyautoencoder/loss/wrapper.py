"""Loss wrappers for easy loss computation and tracking."""
from dataclasses import dataclass
from typing import Dict, Optional, Union
import math
import torch

from .base import LikelihoodType, log_likelihood
from .vae import compute_ELBO
from ..models.autoencoder import AEOutput
from ..models.variational.vae import VAEOutput

LN2 = math.log(2.0)
LOG_2PI = math.log(2.0 * math.pi)  # for Gaussian σ²=1 diagnostics

@dataclass
class LossComponents:
    """
    Container for loss components with detailed metrics.

    Args:
        total (torch.Tensor): Scalar loss to optimize (already reduced over batch).
        components (Dict[str, torch.Tensor]): Named scalar terms that compose the loss
            (e.g., 'log_likelihood', 'kl_divergence').
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
        Loss function for Variational Autoencoders (β-VAE style).
        The optimized loss is the negative ELBO. 
        Uses negative log-likelihood (NLL) of reconstructions:
        - Gaussian (σ²=1): per-dim NLL = 0.5·[(x − x_hat)² + log(2π)].
        - Bernoulli (logits): per-dim NLL = BCEWithLogits(x_hat, x).

        Args:
            beta (float): Weighting factor for the KL term (β-VAE).
            likelihood (Union[str, LikelihoodType]): Likelihood model for p(x|z).
                Supported: 'gaussian' (σ²=1) or 'bernoulli' (logits).
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
                    * 'log_likelihood': Scalar, batch-mean E_q[log p(x|z)] (nats).
                    * 'kl_divergence': Scalar, batch-mean KL(q||p) (nats).
                - metrics (Dict[str, torch.Tensor]):
                    * 'elbo': Scalar, batch-mean ELBO (nats).
                    * 'log_likelihood_per_dim_nats': Scalar, E_q[log p(x|z)] / D_x (nats/dim, negative).
                    * 'nll_per_dim_nats': Scalar, -E_q[log p(x|z)] / D_x (nats/dim, positive).
                    * 'bpd': Scalar, bits per dimension = nll_per_dim_nats / ln(2) (bits/dim).
                    * 'kl_per_latent_dim_nats': Scalar, KL / D_z (nats per latent dim).
                    * 'kl_bits_per_latent_dim': Scalar, KL / (D_z * ln(2)) (bits per latent dim).
                    * 'mse_per_dim' (optional): Scalar, derived from Gaussian σ²=1 identity.

        Notes:
            - Reductions follow: sum over feature dimensions → mean over MC samples (if any) → mean over batch.
            - log p(x|z) is computed by `log_likelihood`:
                * Gaussian (σ²=1): per-dim NLL = 0.5·MSE + 0.5·log(2π).
                * Bernoulli (logits): per-dim NLL = BCEWithLogits.
            - For Gaussian (σ²=1), 'mse_per_dim' is computed via:
                MSE_per_dim = 2·NLL_per_dim − log(2π), clamped to be ≥ 0.
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
        log_like_per_dim_nats = elbo_components.log_likelihood / D_x          # negative
        nll_per_dim = -log_like_per_dim_nats                                  # nats/dim (positive)
        bpd = nll_per_dim / LN2                                               # bits/dim

        kl_per_latent_dim_nats = elbo_components.kl_divergence / D_z          # nats/latent-dim
        kl_bits_per_latent_dim = kl_per_latent_dim_nats / LN2                 # bits/latent-dim

        metrics: Dict[str, torch.Tensor] = {
            'elbo': elbo_components.elbo,
            'log_likelihood_per_dim_nats': log_like_per_dim_nats.detach().cpu(),
            'nll_per_dim_nats': nll_per_dim.detach().cpu(),
            'bpd': bpd.detach().cpu(),
            'kl_per_latent_dim_nats': kl_per_latent_dim_nats.detach().cpu(),
            'kl_bits_per_latent_dim': kl_bits_per_latent_dim.detach().cpu(),
        }

        # Extra: derive MSE/dim for Gaussian(σ²=1)
        if self.likelihood == LikelihoodType.GAUSSIAN:
            # NLL_per_dim = 0.5*MSE_per_dim + 0.5*log(2π) ⇒ MSE_per_dim = 2*NLL_per_dim − log(2π)
            mse_per_dim = torch.clamp(2.0 * nll_per_dim - LOG_2PI, min=0.0)
            metrics['mse_per_dim'] = mse_per_dim

        return LossComponents(
            total=-elbo_components.elbo,  # minimize negative ELBO
            components={
                'log_likelihood': elbo_components.log_likelihood,
                'kl_divergence': elbo_components.kl_divergence,
            },
            metrics=metrics,
        )

class AELoss(BaseLoss):
    def __init__(self, likelihood: Union[str, LikelihoodType] = LikelihoodType.GAUSSIAN):
        """
        Loss function for standard Autoencoders.

        Uses negative log-likelihood (NLL) of reconstructions:
            - Gaussian (σ²=1): per-dim NLL = 0.5·[(x − x_hat)² + log(2π)].
            - Bernoulli (logits): per-dim NLL = BCEWithLogits(x_hat, x).

        Args:
            likelihood (Union[str, LikelihoodType]): Likelihood model for p(x|z).
                Supported: 'gaussian' (σ²=1) or 'bernoulli' (logits).
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
                    * 'reconstruction': Scalar, same as total.
                - metrics (Dict[str, torch.Tensor]):
                    * 'reconstruction_per_dim_nats': Scalar, NLL / D_x (nats/dim, positive).
                    * 'bpd': Scalar, bits per dimension = (NLL / D_x) / ln(2) (bits/dim).
                    * 'mse_per_dim' (optional): Scalar, derived for Gaussian σ²=1.

        Notes:
            - Reductions follow: elementwise log-likelihood → sum over feature dimensions
              → mean over batch.
            - For Gaussian (σ²=1), 'mse_per_dim' is computed via:
                MSE_per_dim = 2·(NLL_per_dim) − log(2π), clamped to be ≥ 0.
            - Ensure inputs match the likelihood’s expected scale:
                * Gaussian: continuous data (typically standardized).
                * Bernoulli: targets in [0, 1], predictions given as logits.
        """
        x_hat = model_output.x_hat

        B = x.size(0)
        D_x = x[0].numel()

        # Elementwise log-likelihood → per-sample sum → batch mean
        ll_elem = log_likelihood(x, x_hat, likelihood=self.likelihood)         # [B, ...]
        ll_per_sample = ll_elem.view(B, -1).sum(-1)                            # [B]
        recon_loss = (-ll_per_sample).mean()                                   # scalar NLL

        # Per-dim diagnostics
        log_like_per_dim_nats = ll_per_sample.mean() / D_x                     # negative
        nll_per_dim = -log_like_per_dim_nats                                   # nats/dim
        bpd = nll_per_dim / LN2                                                # bits/dim

        metrics: Dict[str, torch.Tensor] = {
            'reconstruction_per_dim_nats': nll_per_dim.detach().cpu(),
            'bpd': bpd.detach().cpu(),
        }

        if self.likelihood == LikelihoodType.GAUSSIAN:
            mse_per_dim = torch.clamp(2.0 * nll_per_dim - LOG_2PI, min=0.0)
            metrics['mse_per_dim'] = mse_per_dim

        return LossComponents(
            total=recon_loss,
            components={'reconstruction': recon_loss},
            metrics=metrics,
        )
