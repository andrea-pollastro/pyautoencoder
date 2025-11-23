import math
import torch
import pytest

from pyautoencoder.loss.base import log_likelihood, LikelihoodType
from pyautoencoder.loss.vae import (
    kl_divergence_gaussian,
    compute_ELBO,
    ELBOComponents,
)

# ============== KL ==============

def test_kl_divergence_gaussian_zero_for_standard_normal():
    # q(z|x) = N(0, I) vs p(z) = N(0, I) => KL = 0
    B, Dz = 4, 3
    mu = torch.zeros(B, Dz)
    log_var = torch.zeros(B, Dz)

    kl = kl_divergence_gaussian(mu, log_var)

    assert kl.shape == (B,)
    assert torch.allclose(kl, torch.zeros(B), atol=1e-7)


def test_kl_divergence_gaussian_matches_analytic_formula():
    # Analytic formula per dim:
    # KL = 0.5 * sum( mu^2 + sigma^2 - 1 - log sigma^2 )
    # Here log_var = log sigma^2
    B, Dz = 5, 7
    torch.manual_seed(0)
    mu = torch.randn(B, Dz)
    log_var = torch.randn(B, Dz)  # can be any real -> sigma^2 = exp(log_var)

    kl = kl_divergence_gaussian(mu, log_var)

    sigma2 = torch.exp(log_var)
    manual = 0.5 * (mu.pow(2) + sigma2 - 1.0 - log_var)
    manual_kl = manual.sum(dim=-1)  # [B]

    assert kl.shape == (B,)
    assert torch.allclose(kl, manual_kl, atol=1e-6, rtol=1e-6)

def test_kl_divergence_gaussian_is_non_negative():
    B, Dz = 10, 4
    torch.manual_seed(1)
    mu = torch.randn(B, Dz)
    log_var = torch.randn(B, Dz)

    kl = kl_divergence_gaussian(mu, log_var)

    # KL should be >= 0, allow tiny negative due to numerical noise
    assert torch.all(kl >= -1e-7)

# ========== ELBO ==========

def test_compute_elbo_gaussian_single_sample_matches_manual_decomposition():
    B, D = 3, 4
    torch.manual_seed(0)
    x = torch.randn(B, D)
    x_hat = torch.randn(B, D)
    mu = torch.randn(B, 2)       # arbitrary latent dim
    log_var = torch.randn(B, 2)

    beta = 1.0

    # Use the implementation
    comps = compute_ELBO(
        x=x,
        x_hat=x_hat,
        mu=mu,
        log_var=log_var,
        likelihood=LikelihoodType.GAUSSIAN,
        beta=beta,
    )

    assert isinstance(comps, ELBOComponents)

    # All outputs should be 0-dim tensors (scalars)
    assert comps.elbo.shape == ()
    assert comps.log_likelihood.shape == ()
    assert comps.beta_kl_divergence.shape == ()

    # Manual decomposition
    # log p(x|z): elementwise -> sum over features -> per-sample [B]
    ll_elem = log_likelihood(x, x_hat, likelihood="gaussian")  # [B, D]
    ll_per_sample = ll_elem.view(B, -1).sum(dim=-1)            # [B]
    E_log_px_z = ll_per_sample  # S=1, so Monte Carlo E is just this

    kl_per_sample = kl_divergence_gaussian(mu, log_var)        # [B]

    elbo_per_sample = E_log_px_z - beta * kl_per_sample        # [B]

    manual_elbo = elbo_per_sample.mean()
    manual_ll_mean = E_log_px_z.mean()
    manual_beta_kl_mean = beta * kl_per_sample.mean()

    assert torch.allclose(comps.elbo, manual_elbo, atol=1e-6, rtol=1e-6)
    assert torch.allclose(comps.log_likelihood, manual_ll_mean, atol=1e-6, rtol=1e-6)
    assert torch.allclose(comps.beta_kl_divergence, manual_beta_kl_mean, atol=1e-6, rtol=1e-6)

def test_compute_elbo_gaussian_multi_sample_matches_manual_mc_estimate():
    B, D, Dz, S = 2, 3, 2, 5
    torch.manual_seed(0)
    x = torch.randn(B, D)
    # x_hat has a sample dimension S: [B, S, D]
    x_hat = torch.randn(B, S, D)
    mu = torch.randn(B, Dz)
    log_var = torch.randn(B, Dz)
    beta = 0.7

    comps = compute_ELBO(
        x=x,
        x_hat=x_hat,
        mu=mu,
        log_var=log_var,
        likelihood="gaussian",
        beta=beta,
    )

    # Manual Monte Carlo:
    # log p(x|z_s): [B, S, D] -> sum over D -> [B, S]
    x_expanded = x.unsqueeze(1)                   # [B, 1, D] for broadcasting
    ll_elem = log_likelihood(x_expanded, x_hat, "gaussian")  # [B, S, D]
    ll_per_sample_per_s = ll_elem.view(B, S, -1).sum(-1)     # [B, S]

    E_log_px_z = ll_per_sample_per_s.mean(dim=1)             # [B]

    kl_per_sample = kl_divergence_gaussian(mu, log_var)      # [B]
    elbo_per_sample = E_log_px_z - beta * kl_per_sample      # [B]

    manual_elbo = elbo_per_sample.mean()
    manual_ll_mean = E_log_px_z.mean()
    manual_beta_kl_mean = beta * kl_per_sample.mean()

    assert torch.allclose(comps.elbo, manual_elbo, atol=1e-6, rtol=1e-6)
    assert torch.allclose(comps.log_likelihood, manual_ll_mean, atol=1e-6, rtol=1e-6)
    assert torch.allclose(comps.beta_kl_divergence, manual_beta_kl_mean, atol=1e-6, rtol=1e-6)

def test_compute_elbo_respects_beta_scaling():
    B, D, Dz = 3, 4, 2
    torch.manual_seed(1)
    x = torch.randn(B, D)
    x_hat = torch.randn(B, D)
    mu = torch.randn(B, Dz)
    log_var = torch.randn(B, Dz)

    beta1 = 1.0
    beta2 = 3.5

    c1 = compute_ELBO(x, x_hat, mu, log_var, likelihood="gaussian", beta=beta1)
    c2 = compute_ELBO(x, x_hat, mu, log_var, likelihood="gaussian", beta=beta2)

    # log-likelihood term should be independent of beta
    assert torch.allclose(c1.log_likelihood, c2.log_likelihood, atol=1e-7)

    # KL scaling:
    # elbo(beta2) = ll_mean - beta2 * kl_mean
    # and beta_kl_divergence is beta * kl_mean
    # So their ratio of beta_kl terms should reflect beta2 / beta1
    assert torch.allclose(
        c2.beta_kl_divergence / c1.beta_kl_divergence,
        torch.tensor(beta2 / beta1, dtype=c1.beta_kl_divergence.dtype),
        atol=1e-6,
        rtol=1e-6,
    )

    # Direct identity: elbo = log_likelihood - beta_kl_divergence
    assert torch.allclose(c1.elbo, c1.log_likelihood - c1.beta_kl_divergence, atol=1e-6)
    assert torch.allclose(c2.elbo, c2.log_likelihood - c2.beta_kl_divergence, atol=1e-6)

def test_compute_elbo_bernoulli_with_zero_kl_reduces_to_reconstruction_term():
    # If KL = 0 (mu=0, log_var=0), ELBO should be exactly the reconstruction term
    B, D = 4, 5
    torch.manual_seed(0)
    # Binary targets
    x = torch.randint(low=0, high=2, size=(B, D)).float()
    logits = torch.randn(B, D)  # decoder outputs are logits

    mu = torch.zeros(B, 3)
    log_var = torch.zeros(B, 3)  # KL will be 0

    comps = compute_ELBO(
        x=x,
        x_hat=logits,
        mu=mu,
        log_var=log_var,
        likelihood=LikelihoodType.BERNOULLI,
        beta=1.0,
    )

    # ELBOComponents
    assert isinstance(comps, ELBOComponents)
    assert comps.elbo.shape == ()
    assert comps.log_likelihood.shape == ()
    assert comps.beta_kl_divergence.shape == ()

    # KL==0 => beta_kl_divergence == 0 and elbo == log_likelihood
    assert torch.allclose(comps.beta_kl_divergence, torch.tensor(0.0, dtype=comps.elbo.dtype), atol=1e-7)
    assert torch.allclose(comps.elbo, comps.log_likelihood, atol=1e-7)

    # Check log_likelihood matches batched average of base log_likelihood
    ll_elem = log_likelihood(x, logits, likelihood="bernoulli")  # [B, D]
    ll_per_sample = ll_elem.view(B, -1).sum(-1)                  # [B]
    manual_ll_mean = ll_per_sample.mean()

    assert torch.allclose(comps.log_likelihood, manual_ll_mean, atol=1e-6, rtol=1e-6)

def test_compute_elbo_bernoulli_multi_sample_broadcasts_x_correctly():
    B, D, S = 2, 3, 4
    torch.manual_seed(0)
    x = torch.randint(0, 2, (B, D)).float()
    logits = torch.randn(B, S, D)  # multi-sample logits: [B, S, D]
    mu = torch.zeros(B, 2)
    log_var = torch.zeros(B, 2)

    comps = compute_ELBO(
        x=x,
        x_hat=logits,
        mu=mu,
        log_var=log_var,
        likelihood="bernoulli",
        beta=0.5,
    )

    # Manual Monte Carlo reconstruction term
    x_expanded = x.unsqueeze(1)             # [B, 1, D]
    x_expanded = x_expanded.expand_as(logits)  # [B, S, D]  <-- IMPORTANT

    ll_elem = log_likelihood(x_expanded, logits, "bernoulli")  # [B, S, D]
    ll_per_sample_per_s = ll_elem.view(B, S, -1).sum(-1)       # [B, S]

    E_log_px_z = ll_per_sample_per_s.mean(dim=1)               # [B]
    manual_ll_mean = E_log_px_z.mean()

    assert torch.allclose(comps.log_likelihood, manual_ll_mean, atol=1e-6, rtol=1e-6)

def test_compute_elbo_produces_scalars_and_backpropagates_to_params():
    B, D, Dz = 3, 4, 2
    torch.manual_seed(0)
    x = torch.randn(B, D)

    # Make x_hat, mu, log_var require grad as if they came from a network
    x_hat = torch.randn(B, D, requires_grad=True)
    mu = torch.randn(B, Dz, requires_grad=True)
    log_var = torch.randn(B, Dz, requires_grad=True)

    comps = compute_ELBO(
        x=x,
        x_hat=x_hat,
        mu=mu,
        log_var=log_var,
        likelihood="gaussian",
        beta=1.0,
    )

    # Scalars
    assert comps.elbo.dim() == 0
    assert comps.log_likelihood.dim() == 0
    assert comps.beta_kl_divergence.dim() == 0

    # Backprop
    comps.elbo.backward()

    assert x_hat.grad is not None
    assert mu.grad is not None
    assert log_var.grad is not None

    # Some gradient entries should be non-zero
    assert torch.any(x_hat.grad != 0)
    assert torch.any(mu.grad != 0)
    assert torch.any(log_var.grad != 0)
