"""Loss functions for variational autoencoders with rigorous mathematical implementations."""
import torch
from typing import Union, NamedTuple

from .base import log_likelihood, LikelihoodType

class ELBOComponents(NamedTuple):
    """Components of the ELBO computation."""
    elbo: torch.Tensor                 # scalar: batch-mean ELBO (with grad)
    log_likelihood: torch.Tensor       # scalar: batch-mean E_q[log p(x|z)]
    beta_kl_divergence: torch.Tensor   # scalar: batch-mean β * KL(q||p)

def kl_divergence_gaussian(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """
    Computes the KL divergence KL(q(z|x) || p(z)) between the approximate
    posterior q(z|x) = N(μ, σ²) and the standard normal prior p(z) = N(0, I).

    Args:
        mu (torch.Tensor): Mean of q(z|x), shape [B, D_z].
        log_var (torch.Tensor): Log-variance of q(z|x), shape [B, D_z].

    Returns:
        torch.Tensor: KL divergence per sample, shape [B].
                      Reduction over latent dimensions is performed inside,
                      but not over the batch.
    """
    return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)

def compute_ELBO(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    likelihood: Union[str, LikelihoodType] = LikelihoodType.GAUSSIAN,
    beta: float = 1.0,
) -> ELBOComponents:
    """
    Computes the Evidence Lower Bound (ELBO) for a Variational Autoencoder
    using the β-VAE formulation.

    Args:
        x (torch.Tensor): Ground truth inputs, shape [B, ...].
        x_hat (torch.Tensor): Reconstructed samples, shape [B, ...] or [B, S, ...],
                              where S is the number of Monte Carlo samples from q(z|x).
        mu (torch.Tensor): Mean of q(z|x), shape [B, D_z].
        log_var (torch.Tensor): Log-variance of q(z|x), shape [B, D_z].
        likelihood (Union[str, LikelihoodType]): Likelihood model for p(x|z).
        beta (float): Weighting factor for the KL term (β-VAE).

    Returns:
        ELBOComponents: NamedTuple containing:
            - elbo (torch.Tensor): Scalar, mean ELBO over the batch.
            - log_likelihood (torch.Tensor): Scalar, mean reconstruction term
              E_q[log p(x|z)] over the batch.
            - beta_kl_divergence (torch.Tensor): Scalar, β * mean KL divergence over the batch.

    Notes:
        - If x_hat has no sample dimension, it is assumed to contain a single sample (S=1).
        - log p(x|z) is computed using the `log_likelihood` function, which already
          includes the Gaussian normalization constant for σ²=1 or the stable BCE for Bernoulli.
        - All outputs are averaged over the batch for reporting and optimization.
    """
    # Ensure a sample dimension S exists -> [B, S, ...]
    if x_hat.dim() == x.dim():
        x_hat = x_hat.unsqueeze(1)  # S = 1
    B, S = x_hat.size(0), x_hat.size(1)

    # log p(x|z): elementwise -> sum over features => [B, S]
    log_px_z = log_likelihood(x.unsqueeze(1), x_hat, likelihood=likelihood)
    log_px_z = log_px_z.reshape(B, S, -1).sum(-1)

    # E_q[log p(x|z)] via Monte Carlo average across S: [B]
    E_log_px_z = log_px_z.mean(dim=1)

    # KL(q||p): [B]
    kl_q_p = kl_divergence_gaussian(mu, log_var)

    # ELBO per sample and batch means (retain grads)
    elbo_per_sample = E_log_px_z - beta * kl_q_p          # [B]
    elbo = elbo_per_sample.mean()                         # scalar
    E_log_px_z_mean = E_log_px_z.mean()                   # scalar
    beta_kl_q_p_mean = beta * kl_q_p.mean()               # scalar

    return ELBOComponents(
        elbo=elbo,
        log_likelihood=E_log_px_z_mean,
        beta_kl_divergence=beta_kl_q_p_mean,
    )
