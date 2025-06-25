import math
import torch
import torch.nn.functional as F

def log_likelihood(x: torch.Tensor, 
                   x_hat: torch.Tensor, 
                   likelihood: str = 'gaussian') -> torch.Tensor:
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
    log_p_x_given_z = log_likelihood(x_exp, x_hat, likelihood=likelihood)
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
