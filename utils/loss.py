from typing import List, Tuple
import math
import torch
import torch.nn.functional as F

def log_likelihood(x: torch.Tensor, 
                   x_hat: torch.Tensor, 
                   likelihood: str = 'gaussian',
                   reduction: str = 'mean') -> torch.Tensor:
    """
    Computes the log-likelihood of the reconstructed tensor x_hat given the original tensor x,
    under either a Bernoulli or Gaussian likelihood assumption with unit variance and i.i.d. samples,
    without applying any reduction.

    Args:
        x (torch.Tensor): Ground truth input tensor.
        x_hat (torch.Tensor): Reconstructed tensor.
        likelihood (str): Type of likelihood model to use: 'bernoulli' or 'gaussian'.

    Returns:
        torch.Tensor: The log-likelihood value (i.e., negative of the appropriate loss function 
                      with normalization constant included for Gaussian).

    Raises:
        ValueError: If the likelihood type is not one of 'bernoulli' or 'gaussian'.

    Notes:
        - Bernoulli likelihood uses binary cross-entropy loss.
        - Gaussian likelihood assumes unit variance and computes:
              log p(x | x_hat) = -0.5 * ||x - x_hat||^2 - (D/2) * log(2pi)
          where D is the number of features per sample.
    """
    likelihood = likelihood.lower()
    if likelihood not in ['bernoulli', 'gaussian']:
        raise ValueError(f"Unknown likelihood: '{likelihood}'. Choose 'bernoulli' or 'gaussian'.")

    if likelihood == 'bernoulli':
        return -F.binary_cross_entropy(x_hat, x, reduction='none')
    
    if likelihood == 'gaussian':
        D = x[0].numel()
        mse = F.mse_loss(x_hat, x, reduction='none')
        norm_constant = 0.5 * D * math.log(2 * math.pi)
        return -0.5 * mse - norm_constant
    
def ELBO(x: torch.Tensor, 
         x_hat: torch.Tensor, 
         mu: torch.Tensor, 
         log_var: torch.Tensor,
         likelihood: str = 'gaussian',
         beta: float = 1.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the Evidence Lower Bound (ELBO) for a Variational Autoencoder.

    Args:
        x (torch.Tensor): Original input tensor of shape [B, ...].
        x_hat (torch.Tensor): Reconstructed samples of shape [B, L, ...], where L is the number of latent samples.
        mu (torch.Tensor): Mean of the approximate posterior q(z|x), shape [B, latent_dim].
        log_var (torch.Tensor): Log-variance of q(z|x), shape [B, latent_dim].
        likelihood (str): Likelihood model to use: 'gaussian' or 'bernoulli'.
        beta (float): Weighting factor for the KL divergence term (used in beta-VAE).

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - ELBO (scalar): The final ELBO estimate averaged over the batch.
            - log_p_x_given_z (scalar): Expected log-likelihood term.
            - kl_divergence (scalar): KL divergence term.

    Notes:
        The reconstruction term is averaged over the latent samples L and the batch.
        The KL divergence is computed assuming a standard normal prior p(z).
    """
    B, L = x_hat.size(0), x_hat.size(1)

    # Log-likelihood E_q[log p(x|z)]
    x_exp = x.unsqueeze(1).expand(-1, L, *([-1] * (x.ndim - 1)))
    log_p_x_given_z = log_likelihood(x_exp, x_hat, likelihood=likelihood, reduction='none')
    log_p_x_given_z = log_p_x_given_z.view(B, L, -1).sum(-1)
    log_p_x_given_z = log_p_x_given_z.mean(dim=1)

    # KL divergence KL(q(z|x) || p(z)) = log q(z|x) - log p(z)
    kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)

    # ELBO
    elbo_per_sample = log_p_x_given_z - beta * kl_divergence

    # Final metrics
    elbo = elbo_per_sample.mean()
    log_p_x_given_z = log_p_x_given_z.mean()
    kl_divergence = kl_divergence.mean()

    return elbo, log_p_x_given_z, kl_divergence

def IWAE_ELBO(x: torch.Tensor, 
              x_hat: torch.Tensor, 
              z: torch.Tensor, 
              mu: torch.Tensor, 
              log_var: torch.Tensor,
              likelihood: str = 'gaussian') -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the IWAE (Importance Weighted Autoencoder) ELBO.

    Args:
        x (torch.Tensor): Original input tensor of shape [B, ...].
        x_hat (torch.Tensor): Reconstructed samples of shape [B, L, ...], where L is the number of latent samples.
        z (torch.Tensor): Latent samples drawn from q(z|x), shape [B, L, latent_dim].
        mu (torch.Tensor): Mean of q(z|x), shape [B, latent_dim].
        log_var (torch.Tensor): Log-variance of q(z|x), shape [B, latent_dim].
        likelihood (str): Likelihood model to use: 'gaussian' or 'bernoulli'.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - IWAE ELBO (scalar): The estimated tighter lower bound.
            - log_p_x_given_z (scalar): Mean reconstruction log-likelihood.
            - kl_divergence (scalar): Mean KL divergence between q(z|x) and p(z).

    Notes:
        - Importance weights are normalized using a softmax-like operation on log-space to improve numerical stability.
        - This implementation assumes a unit-variance standard normal prior p(z).
    """
    B, L = x_hat.size(0), x_hat.size(1)
    x_exp = x.unsqueeze(1).expand(B, L, *([-1] * (x.ndim - 1)))

    # Log-likelihood p(x|z_i)
    log_p_x_given_z = log_likelihood(x_exp, x_hat, likelihood=likelihood, reduction='none')
    log_p_x_given_z = log_p_x_given_z.view(B, L, -1).sum(-1)

    # Prior p(z_i)
    log_2pi = torch.log(torch.tensor(2 * math.pi, device=z.device))
    log_p_z = -0.5 * (z.pow(2) + log_2pi).sum(-1) # NOTE the sum across the dimensions, ok for normalization constant

    # Posterior q(z_i|x)
    mu_expanded = mu.unsqueeze(1).expand(B, L, *([-1] * (mu.ndim - 1)))
    log_var_expanded = log_var.unsqueeze(1).expand(B, L, *([-1] * (log_var.ndim - 1)))
    log_q_z_given_x = -0.5 * (((z - mu_expanded)**2) / log_var_expanded.exp() + log_var_expanded + log_2pi).sum(-1)

    # Importance weights
    # Note: log w_i = log(p(x,z_i)/q(z_i|x)) = log p(x|z_i) + log p(z_i) - log q(z_i|x)
    log_w = log_p_x_given_z + log_p_z - log_q_z_given_x

    # IWAE bound
    iwae_elbo_per_sample = torch.logsumexp(log_w, dim=1) - math.log(L)

    # Final metrics
    elbo = iwae_elbo_per_sample.mean()
    log_p_x_given_z = log_p_x_given_z.mean()
    kl_divergence = (log_q_z_given_x - log_p_z).mean()

    return elbo, log_p_x_given_z, kl_divergence

def hierarchical_ELBO(x: torch.Tensor,
                      x_hat: torch.Tensor,
                      mus: List[torch.Tensor],
                      log_vars: List[torch.Tensor],
                      likelihood: str = 'gaussian',
                      beta: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the ELBO for a Hierarchical VAE.

    Args:
        x (torch.Tensor): Original input of shape [B, ...].
        x_hat (torch.Tensor): Reconstructed samples from z_1, shape [B, L, ...].
        mus (List[torch.Tensor]): List of means from each q(z_l | ·), each shape [B, latent_dim_l].
        log_vars (List[torch.Tensor]): List of log-variances from each q(z_l | ·), shape [B, latent_dim_l].
        likelihood (str): Type of likelihood for reconstruction ('gaussian' or 'bernoulli').
        beta (float): Weighting factor for total KL term (useful for beta-VAE).

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - ELBO (scalar): Final ELBO estimate averaged over the batch.
            - log_p_x_given_z (scalar): Expected log-likelihood term.
            - kl_total (scalar): Total KL divergence across all latent layers.

    Notes:
        Assumes standard normal priors for all latent variables.
    """
    B, L = x_hat.shape[:2]

    # Reconstruction term: E_q(z_1)[log p(x | z_1)]
    x_exp = x.unsqueeze(1).expand_as(x_hat)
    log_p_x_given_z = log_likelihood(x_exp, x_hat, likelihood=likelihood, reduction='none')
    log_p_x_given_z = log_p_x_given_z.view(B, L, -1).sum(-1)  # [B, L]
    log_p_x_given_z = log_p_x_given_z.mean(dim=1)  # [B]

    # KL terms for each layer
    kl_divergences = [
        -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp(), dim=-1)  # [B]
        for mu, lv in zip(mus, log_vars)
    ]
    kl_total = sum(kl_divergences)  # still shape [B]

    # ELBO
    elbo_per_sample = log_p_x_given_z - beta * kl_total
    elbo = elbo_per_sample.mean()
    log_p_x_given_z = log_p_x_given_z.mean()
    kl_total = kl_total.mean()

    return elbo, log_p_x_given_z, kl_total
