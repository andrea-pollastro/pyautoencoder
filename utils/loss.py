import math
import torch
import torch.nn.functional as F

def log_likelihood(x: torch.Tensor, 
                   x_hat: torch.Tensor, 
                   likelihood: str = 'gaussian',
                   reduction: str = 'mean') -> torch.Tensor:
    likelihood = likelihood.lower()
    if likelihood not in ['bernoulli', 'gaussian']:
        raise ValueError(f"Unknown likelihood: '{likelihood}'. Choose 'bernoulli' or 'gaussian'.")

    if likelihood == 'bernoulli':
        return -F.binary_cross_entropy(x_hat, x, reduction=reduction)
    if likelihood == 'gaussian':
        D = x[0].numel()
        mse = F.mse_loss(x_hat, x, reduction=reduction)
        norm_constant = 0.5 * D * math.log(2 * math.pi)
        return -0.5 * mse - norm_constant
    
def ELBO(x: torch.Tensor, 
         x_hat: torch.Tensor, 
         mu: torch.Tensor, 
         log_var: torch.Tensor,
         likelihood: str = 'gaussian',
         beta: float = 1.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, L = x_hat.size(0), x_hat.size(1)

    # Expand x to match the number of samples L
    x_expanded = x.unsqueeze(1).expand(-1, L, *([-1] * (x.ndim - 1)))

    # Log-likelihood
    log_p_x_given_z = log_likelihood(x_expanded, x_hat, likelihood=likelihood, reduction='none')
    log_p_x_given_z = log_p_x_given_z.view(B, L, -1).sum(-1)
    log_p_x_given_z = log_p_x_given_z.mean(dim=1)  # Average over samples

    # KL divergence
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
              mu: torch.Tensor, 
              log_var: torch.Tensor,
              likelihood: str = 'gaussian') -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # TODO continue from here
    raise NotImplementedError("IWAE ELBO is not implemented yet.")