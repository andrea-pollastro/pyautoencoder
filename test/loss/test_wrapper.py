import math
import torch
import pytest

from pyautoencoder.loss.base import log_likelihood, LikelihoodType
from pyautoencoder.loss.vae import compute_ELBO
from pyautoencoder.loss.wrapper import (
    VAELoss,
    AELoss,
    LossComponents,
)

LN2 = math.log(2.0)
LOG_2PI = math.log(2.0 * math.pi)

class DummyAEOutput:
    def __init__(self, x_hat: torch.Tensor, z: torch.Tensor | None = None):
        self.x_hat = x_hat
        self.z = z

class DummyVAEOutput:
    def __init__(self, x_hat: torch.Tensor, z: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor):
        self.x_hat = x_hat
        self.z = z
        self.mu = mu
        self.log_var = log_var

def test_losscomponents_holds_total_components_and_metrics():
    total = torch.tensor(1.23)
    components = {"a": torch.tensor(0.5), "b": torch.tensor(0.73)}
    metrics = {"m1": torch.tensor(10.0)}

    lc = LossComponents(total=total, components=components, metrics=metrics)

    assert lc.total is total
    assert lc.components is components
    assert lc.metrics is metrics

def test_aeloss_gaussian_matches_manual_nll_and_metrics():
    B, D = 3, 4
    torch.manual_seed(0)
    x = torch.randn(B, D)
    x_hat = torch.randn(B, D, requires_grad=True)

    ae_out = DummyAEOutput(x_hat=x_hat)

    loss_fn = AELoss(likelihood=LikelihoodType.GAUSSIAN)
    loss = loss_fn(x, ae_out) # type: ignore

    assert isinstance(loss, LossComponents)
    assert loss.total.shape == ()

    # Manual computation using log_likelihood (already tested separately)
    ll_elem = log_likelihood(x, x_hat, likelihood="gaussian")   # [B, D]
    ll_per_sample = ll_elem.view(B, -1).sum(-1)                 # [B]
    manual_nll = (-ll_per_sample).mean()                        # scalar

    assert torch.allclose(loss.total, manual_nll, atol=1e-6)

    # components
    assert "negative_log_likelihood" in loss.components
    assert torch.allclose(loss.components["negative_log_likelihood"], manual_nll, atol=1e-6)

    # metrics
    D_x = x[0].numel()
    nll_per_dim_nats_manual = manual_nll / D_x
    nll_per_dim_bits_manual = nll_per_dim_nats_manual / LN2

    m = loss.metrics
    assert m is not None

    assert torch.allclose(m["nll_per_dim_nats"], nll_per_dim_nats_manual.cpu(), atol=1e-6)
    assert torch.allclose(m["nll_per_dim_bits"], nll_per_dim_bits_manual.cpu(), atol=1e-6)

    # mse_per_dim should exist and follow the identity:
    # MSE_per_dim = 2*NLL_per_dim - log(2π)
    assert "mse_per_dim" in m
    mse_per_dim_manual = max(0.0, float(2.0 * nll_per_dim_nats_manual - LOG_2PI))
    assert torch.allclose(m["mse_per_dim"], torch.tensor(mse_per_dim_manual), atol=1e-6)

    # metrics must be detached & on CPU
    for v in m.values():
        assert v.requires_grad is False
        assert v.device.type == "cpu"

    # total should still carry grad w.r.t. x_hat
    loss.total.backward()
    assert x_hat.grad is not None
    assert torch.any(x_hat.grad != 0)

def test_aeloss_gaussian_perfect_reconstruction_gives_mse_zero():
    B, D = 2, 3
    x = torch.randn(B, D)
    x_hat = x.clone().detach().requires_grad_(True)

    ae_out = DummyAEOutput(x_hat=x_hat)
    loss_fn = AELoss(likelihood=LikelihoodType.GAUSSIAN)
    loss = loss_fn(x, ae_out) # type: ignore

    m = loss.metrics
    assert m is not None

    # In Gaussian sigma^2=1, if x_hat == x, per-dim NLL = 0.5*log(2π)
    D_x = x[0].numel()
    nll_per_dim_nats = m["nll_per_dim_nats"].item()
    assert torch.allclose(
        torch.tensor(nll_per_dim_nats),
        torch.tensor(0.5 * LOG_2PI),
        atol=1e-5,
    )

    # mse_per_dim ≈ 0 (clamped non-negative)
    assert torch.allclose(m["mse_per_dim"], torch.tensor(0.0), atol=1e-6)

def test_aeloss_bernoulli_matches_manual_nll_and_has_no_mse_metric():
    B, D = 3, 4
    torch.manual_seed(1)
    x = torch.randint(0, 2, (B, D)).float()
    logits = torch.randn(B, D, requires_grad=True)

    ae_out = DummyAEOutput(x_hat=logits)

    loss_fn = AELoss(likelihood="bernoulli")
    loss = loss_fn(x, ae_out) # type: ignore

    assert isinstance(loss, LossComponents)

    ll_elem = log_likelihood(x, logits, "bernoulli")        # [B, D]
    ll_per_sample = ll_elem.view(B, -1).sum(-1)             # [B]
    manual_nll = (-ll_per_sample).mean()

    assert torch.allclose(loss.total, manual_nll, atol=1e-6)
    assert torch.allclose(
        loss.components["negative_log_likelihood"], manual_nll, atol=1e-6
    )

    # metrics
    D_x = x[0].numel()
    nll_per_dim_nats_manual = manual_nll / D_x
    nll_per_dim_bits_manual = nll_per_dim_nats_manual / LN2

    m = loss.metrics
    assert m is not None
    assert torch.allclose(m["nll_per_dim_nats"], nll_per_dim_nats_manual.cpu(), atol=1e-6)
    assert torch.allclose(m["nll_per_dim_bits"], nll_per_dim_bits_manual.cpu(), atol=1e-6)

    # For Bernoulli, 'mse_per_dim' should NOT be present
    assert "mse_per_dim" not in m

    # Grad check
    loss.total.backward()
    assert logits.grad is not None
    assert torch.any(logits.grad != 0)

def test_vaeloss_gaussian_matches_compute_elbo_and_metrics():
    B, D, Dz, S = 3, 4, 2, 5
    torch.manual_seed(0)

    x = torch.randn(B, D)
    # x_hat from VAE: [B, S, D]
    x_hat = torch.randn(B, S, D, requires_grad=True)
    mu = torch.randn(B, Dz, requires_grad=True)
    log_var = torch.randn(B, Dz, requires_grad=True)

    vae_out = DummyVAEOutput(x_hat=x_hat, z=torch.randn(B, S, Dz), mu=mu, log_var=log_var)

    beta = 1.7
    loss_fn = VAELoss(beta=beta, likelihood=LikelihoodType.GAUSSIAN)
    loss = loss_fn(x, vae_out) # type: ignore

    assert isinstance(loss, LossComponents)

    # Use compute_ELBO directly
    elbo_comp = compute_ELBO(
        x=x,
        x_hat=x_hat,
        mu=mu,
        log_var=log_var,
        likelihood="gaussian",
        beta=beta,
    )

    # total = -ELBO
    assert torch.allclose(loss.total, -elbo_comp.elbo, atol=1e-6)

    # components
    assert torch.allclose(
        loss.components["negative_log_likelihood"],
        -elbo_comp.log_likelihood,
        atol=1e-6,
    )
    assert torch.allclose(
        loss.components["beta_kl_divergence"],
        elbo_comp.beta_kl_divergence,
        atol=1e-6,
    )

    # total should equal sum of components
    assert torch.allclose(
        loss.total,
        loss.components["negative_log_likelihood"] + loss.components["beta_kl_divergence"],
        atol=1e-6,
    )

    # metrics — per-dim normalization
    D_x = x[0].numel()
    D_z = mu.size(-1)

    nll_per_dim_nats_manual = -elbo_comp.log_likelihood / D_x
    nll_per_dim_bits_manual = nll_per_dim_nats_manual / LN2
    beta_kl_per_latent_dim_nats_manual = elbo_comp.beta_kl_divergence / D_z
    beta_kl_per_latent_dim_bits_manual = beta_kl_per_latent_dim_nats_manual / LN2

    m = loss.metrics
    assert m is not None

    assert torch.allclose(m["elbo"], elbo_comp.elbo.detach().cpu(), atol=1e-6)
    assert torch.allclose(m["nll_per_dim_nats"], nll_per_dim_nats_manual.detach().cpu(), atol=1e-6)
    assert torch.allclose(m["nll_per_dim_bits"], nll_per_dim_bits_manual.detach().cpu(), atol=1e-6)
    assert torch.allclose(
        m["beta_kl_per_latent_dim_nats"],
        beta_kl_per_latent_dim_nats_manual.detach().cpu(),
        atol=1e-6,
    )
    assert torch.allclose(
        m["beta_kl_per_latent_dim_bits"],
        beta_kl_per_latent_dim_bits_manual.detach().cpu(),
        atol=1e-6,
    )

    # mse_per_dim present for Gaussian
    assert "mse_per_dim" in m

    # metrics detached & CPU
    for v in m.values():
        assert v.requires_grad is False
        assert v.device.type == "cpu"

    # Backprop: gradients should flow through x_hat, mu, log_var
    loss.total.backward()
    assert x_hat.grad is not None and torch.any(x_hat.grad != 0)
    assert mu.grad is not None and torch.any(mu.grad != 0)
    assert log_var.grad is not None and torch.any(log_var.grad != 0)

def test_vaeloss_gaussian_perfect_reconstruction_and_zero_kl():
    B, D, Dz, S = 2, 3, 4, 1
    torch.manual_seed(0)

    x = torch.randn(B, D)
    x_hat = x.unsqueeze(1).clone().detach().requires_grad_(True)  # [B, 1, D]

    mu = torch.zeros(B, Dz, requires_grad=True)
    log_var = torch.zeros(B, Dz, requires_grad=True)

    vae_out = DummyVAEOutput(x_hat=x_hat, z=torch.zeros(B, S, Dz), mu=mu, log_var=log_var)

    loss_fn = VAELoss(beta=1.0, likelihood="gaussian")
    loss = loss_fn(x, vae_out) # type: ignore

    m = loss.metrics
    assert m is not None

    # KL = 0, so ELBO = E[log p(x|z)] only
    # With perfect reconstruction, per-dim NLL = 0.5*log(2π)
    nll_per_dim_nats = m["nll_per_dim_nats"].item()
    assert torch.allclose(
        torch.tensor(nll_per_dim_nats),
        torch.tensor(0.5 * LOG_2PI),
        atol=1e-5,
    )

    # mse_per_dim should be ~0
    assert torch.allclose(m["mse_per_dim"], torch.tensor(0.0), atol=1e-6)

def test_vaeloss_bernoulli_zero_kl_reduces_to_reconstruction_term():
    B, D, Dz, S = 4, 5, 3, 2
    torch.manual_seed(0)

    x = torch.randint(0, 2, (B, D)).float()
    logits = torch.randn(B, S, D, requires_grad=True)
    mu = torch.zeros(B, Dz, requires_grad=True)
    log_var = torch.zeros(B, Dz, requires_grad=True)

    vae_out = DummyVAEOutput(x_hat=logits, z=torch.zeros(B, S, Dz), mu=mu, log_var=log_var)

    beta = 2.0
    loss_fn = VAELoss(beta=beta, likelihood="bernoulli")
    loss = loss_fn(x, vae_out) # type: ignore

    elbo_comp = compute_ELBO(
        x=x,
        x_hat=logits,
        mu=mu,
        log_var=log_var,
        likelihood="bernoulli",
        beta=beta,
    )

    # With mu=0, log_var=0, KL = 0, so ELBO == E[log p(x|z)]
    # total = -ELBO = -log_likelihood
    assert torch.allclose(loss.total, -elbo_comp.elbo, atol=1e-6)
    assert torch.allclose(loss.components["beta_kl_divergence"], elbo_comp.beta_kl_divergence, atol=1e-6)
    assert torch.allclose(loss.components["beta_kl_divergence"], torch.tensor(0.0), atol=1e-7)

    # No mse_per_dim for Bernoulli
    assert "mse_per_dim" not in (loss.metrics or {})

def test_vaeloss_beta_scaling_behaviour():
    B, D, Dz, S = 3, 4, 2, 3
    torch.manual_seed(2)

    x = torch.randn(B, D)
    x_hat = torch.randn(B, S, D)
    mu = torch.randn(B, Dz)
    log_var = torch.randn(B, Dz)

    vae_out = DummyVAEOutput(x_hat=x_hat, z=torch.randn(B, S, Dz), mu=mu, log_var=log_var)

    beta1 = 1.0
    beta2 = 4.0

    loss1 = VAELoss(beta=beta1, likelihood="gaussian")(x, vae_out) # type: ignore
    loss2 = VAELoss(beta=beta2, likelihood="gaussian")(x, vae_out) # type: ignore

    # NLL term independent of beta
    assert torch.allclose(
        loss1.components["negative_log_likelihood"],
        loss2.components["negative_log_likelihood"],
        atol=1e-6,
    )

    # beta_kl_divergence scales linearly with beta
    ratio = loss2.components["beta_kl_divergence"] / loss1.components["beta_kl_divergence"]
    assert torch.allclose(ratio, torch.tensor(beta2 / beta1, dtype=ratio.dtype), atol=1e-6)

    # total = NLL + beta * KL
    for L in (loss1, loss2):
        assert torch.allclose(
            L.total,
            L.components["negative_log_likelihood"] + L.components["beta_kl_divergence"],
            atol=1e-6,
        )
