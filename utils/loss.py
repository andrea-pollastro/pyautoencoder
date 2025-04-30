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

    # Log-likelihood p(x|z)
    x_expanded = x.unsqueeze(1).expand(-1, L, *([-1] * (x.ndim - 1)))
    log_p_x_given_z = log_likelihood(x_expanded, x_hat, likelihood=likelihood, reduction='none')
    log_p_x_given_z = log_p_x_given_z.view(B, L, -1).sum(-1)
    log_p_x_given_z = log_p_x_given_z.mean(dim=1)  # Average over samples

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
    B, L = x_hat.size(0), x_hat.size(1)
    x_expanded = x.unsqueeze(1).expand(B, L, *([-1] * (x.ndim - 1)))
    log_2pi = torch.log(torch.tensor(2 * math.pi, device=z.device))

    # Log-likelihood p(x|z_i)
    log_p_x_given_z = log_likelihood(x_expanded, x_hat, likelihood=likelihood, reduction='none')
    log_p_x_given_z = log_p_x_given_z.view(B, L, -1).sum(-1)

    # Prior p(z_i)
    log_p_z = -0.5 * (z.pow(2) + log_2pi).sum(-1)

    # Posterior q(z_i|x)
    mu_expanded = mu.unsqueeze(1).expand(B, L, *([-1] * (mu.ndim - 1)))
    log_var_expanded = log_var.unsqueeze(1).expand(B, L, *([-1] * (log_var.ndim - 1)))
    log_q_z_given_x = -0.5 * (((z - mu_expanded)**2) / log_var_expanded.exp() + log_var_expanded + log_2pi).sum(-1)

    # Importance weights
    # Note: log w_i = log(p(x,z_i)/q(z_i|x)) = log p(x|z_i) + log p(z_i) - log q(z_i|x)
    log_w = log_p_x_given_z + log_p_z - log_q_z_given_x

    # Normalized importance weights
    log_w_tilde = log_w - torch.logsumexp(log_w, dim=1, keepdim=True)
    w_tilde = log_w_tilde.exp().detach()
    
    iwae_elbo_per_sample = (w_tilde * log_w).sum(-1)

    # Final metrics
    elbo = iwae_elbo_per_sample.mean()
    log_p_x_given_z = log_p_x_given_z.mean()
    kl_divergence = (log_q_z_given_x - log_p_z).mean()

    return elbo, log_p_x_given_z, kl_divergence