import pytest
import torch
import torch.nn as nn
import math

from pyautoencoder.variational.stochastic_layers import FullyFactorizedGaussian

# ======================= FullyFactorizedGaussian =======================

def test_ffg_initial_state():
    latent_dim = 5
    head = FullyFactorizedGaussian(latent_dim=latent_dim)

    assert head.latent_dim == latent_dim
    # mu and log_var are lazy linear layers, not yet materialized
    assert isinstance(head.mu, nn.LazyLinear)
    assert isinstance(head.log_var, nn.LazyLinear)
    assert head.mu.has_uninitialized_params()
    assert head.log_var.has_uninitialized_params()


def test_ffg_lazy_init_materializes_on_first_forward():
    latent_dim = 4
    B, F = 3, 7
    x = torch.randn(B, F)

    head = FullyFactorizedGaussian(latent_dim=latent_dim)
    head.eval()

    # Before first forward: mu and log_var are still LazyLinear
    assert isinstance(head.mu, nn.LazyLinear)
    assert isinstance(head.log_var, nn.LazyLinear)

    z, mu, log_var = head(x)

    # After first forward: LazyLinear has materialized into a regular Linear
    assert not isinstance(head.mu, nn.LazyLinear)
    assert not isinstance(head.log_var, nn.LazyLinear)
    assert isinstance(head.mu, nn.Linear)
    assert head.mu.in_features == F
    assert head.log_var.in_features == F
    assert head.mu.out_features == latent_dim
    assert head.log_var.out_features == latent_dim

    assert z.shape == (B, 1, latent_dim)
    assert mu.shape == (B, latent_dim)
    assert log_var.shape == (B, latent_dim)


def test_ffg_training_forward_shapes_and_grad_and_sampling():
    B, F, Dz, S = 4, 6, 3, 5
    x = torch.randn(B, F)

    head = FullyFactorizedGaussian(latent_dim=Dz)

    head.train()
    torch.set_grad_enabled(True)

    z, mu, log_var = head(x, S=S)

    assert z.shape == (B, S, Dz)
    assert mu.shape == (B, Dz)
    assert log_var.shape == (B, Dz)

    assert z.requires_grad is True
    assert mu.requires_grad is True
    assert log_var.requires_grad is True

    std = torch.exp(0.5 * log_var)
    assert torch.isfinite(std).all()


def test_ffg_training_sampling_uses_global_rng():
    """Two calls with the same seed should produce identical samples."""
    B, F, Dz, S = 3, 5, 2, 4
    x = torch.randn(B, F)

    head = FullyFactorizedGaussian(latent_dim=Dz)
    head.train()
    torch.set_grad_enabled(True)

    # Warm up to materialize lazy layers before seeded calls, so that
    # lazy-init RNG consumption doesn't interfere with the seed.
    head(x, S=1)

    torch.manual_seed(42)
    z1, mu1, log_var1 = head(x, S=S)

    torch.manual_seed(42)
    z2, mu2, log_var2 = head(x, S=S)

    assert torch.allclose(mu1, mu2)
    assert torch.allclose(log_var1, log_var2)
    assert torch.allclose(z1, z2)


def test_ffg_eval_forward_is_deterministic_and_uses_mu_only():
    B, F, Dz, S = 3, 5, 4, 6
    x = torch.randn(B, F)

    head = FullyFactorizedGaussian(latent_dim=Dz)

    head.eval()
    torch.set_grad_enabled(True)

    z, mu, log_var = head(x, S=S)

    assert z.shape == (B, S, Dz)
    assert mu.shape == (B, Dz)
    assert log_var.shape == (B, Dz)

    expected_z = mu.unsqueeze(1).expand(-1, S, -1)
    assert torch.allclose(z, expected_z)

    assert z.requires_grad is True
    assert mu.requires_grad is True
    assert log_var.requires_grad is True

def test_ffg_eval_forward_respects_default_S_equals_1():
    B, F, Dz = 2, 4, 3
    x = torch.randn(B, F)

    head = FullyFactorizedGaussian(latent_dim=Dz)
    head.eval()

    z, mu, log_var = head(x)  # S default = 1

    assert z.shape == (B, 1, Dz)
    assert mu.shape == (B, Dz)
    assert log_var.shape == (B, Dz)
    expected_z = mu.unsqueeze(1)
    assert torch.allclose(z, expected_z)


def test_ffg_rejects_invalid_S_values():
    B, F, Dz = 2, 4, 3
    x = torch.randn(B, F)

    head = FullyFactorizedGaussian(latent_dim=Dz)
    head.train()

    with pytest.raises(ValueError, match="S must be >= 1"):
        head(x, S=0)

    with pytest.raises(ValueError, match="S must be >= 1"):
        head(x, S=-1)

    with pytest.raises(ValueError, match="S must be >= 1"):
        head(x, S=-5)

# ======================= FullyFactorizedGaussian._reparametrize =======================

def test__reparametrize_shapes():
    B, Dz, S = 3, 5, 4
    mu = torch.randn(B, Dz, requires_grad=True)
    log_var = torch.randn(B, Dz, requires_grad=True)

    head = FullyFactorizedGaussian(latent_dim=Dz)
    z = head._reparametrize(mu=mu, log_var=log_var, S=S)

    assert z.shape == (B, S, Dz)
    assert z.requires_grad is True


def test__reparametrize_deterministic_with_seed():
    B, Dz, S = 2, 3, 4
    mu = torch.randn(B, Dz)
    log_var = torch.randn(B, Dz)

    head = FullyFactorizedGaussian(latent_dim=Dz)

    torch.manual_seed(42)
    z1 = head._reparametrize(mu=mu, log_var=log_var, S=S)

    torch.manual_seed(42)
    z2 = head._reparametrize(mu=mu, log_var=log_var, S=S)

    assert torch.allclose(z1, z2)


def test__reparametrize_stochastic_without_seed():
    B, Dz, S = 2, 3, 10
    mu = torch.randn(B, Dz)
    log_var = torch.randn(B, Dz)

    head = FullyFactorizedGaussian(latent_dim=Dz)

    z1 = head._reparametrize(mu=mu, log_var=log_var, S=S)
    z2 = head._reparametrize(mu=mu, log_var=log_var, S=S)

    assert not torch.allclose(z1, z2)


def test__reparametrize_preserves_dtype():
    B, Dz, S = 2, 3, 4

    mu32 = torch.randn(B, Dz, dtype=torch.float32)
    log_var32 = torch.randn(B, Dz, dtype=torch.float32)
    head = FullyFactorizedGaussian(latent_dim=Dz)
    z32 = head._reparametrize(mu=mu32, log_var=log_var32, S=S)
    assert z32.dtype == torch.float32

    mu64 = torch.randn(B, Dz, dtype=torch.float64)
    log_var64 = torch.randn(B, Dz, dtype=torch.float64)
    z64 = head._reparametrize(mu=mu64, log_var=log_var64, S=S)
    assert z64.dtype == torch.float64


def test__reparametrize_mean_close_to_mu():
    B, Dz, S = 2, 3, 10000
    mu = torch.randn(B, Dz)
    log_var = torch.zeros(B, Dz)

    head = FullyFactorizedGaussian(latent_dim=Dz)
    z = head._reparametrize(mu=mu, log_var=log_var, S=S)

    z_mean = z.mean(dim=1)  # [B, Dz]
    assert torch.allclose(z_mean, mu, atol=0.05)


def test__reparametrize_std_close_to_expected():
    B, Dz, S = 2, 3, 10000
    mu = torch.zeros(B, Dz)
    expected_std = 2.0
    log_var = torch.full((B, Dz), 2.0 * math.log(expected_std))

    head = FullyFactorizedGaussian(latent_dim=Dz)
    z = head._reparametrize(mu=mu, log_var=log_var, S=S)

    z_std = z.std(dim=1)  # [B, Dz]
    assert torch.allclose(z_std, torch.full_like(z_std, expected_std), atol=0.1)


def test__reparametrize_backward():
    B, Dz, S = 2, 3, 4
    mu = torch.randn(B, Dz, requires_grad=True)
    log_var = torch.randn(B, Dz, requires_grad=True)

    head = FullyFactorizedGaussian(latent_dim=Dz)
    z = head._reparametrize(mu=mu, log_var=log_var, S=S)

    loss = z.sum()
    loss.backward()

    assert mu.grad is not None
    assert log_var.grad is not None
    assert torch.any(mu.grad != 0)
    assert torch.any(log_var.grad != 0)


def test__reparametrize_with_s_equals_one():
    B, Dz = 3, 4
    mu = torch.randn(B, Dz)
    log_var = torch.randn(B, Dz)

    head = FullyFactorizedGaussian(latent_dim=Dz)
    z = head._reparametrize(mu=mu, log_var=log_var, S=1)

    assert z.shape == (B, 1, Dz)


def test__reparametrize_large_s():
    B, Dz, S = 2, 3, 1000
    mu = torch.randn(B, Dz)
    log_var = torch.randn(B, Dz)

    head = FullyFactorizedGaussian(latent_dim=Dz)
    z = head._reparametrize(mu=mu, log_var=log_var, S=S)

    assert z.shape == (B, S, Dz)
    assert torch.isfinite(z).all()
