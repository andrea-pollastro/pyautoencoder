import pytest
import torch
import torch.nn as nn

from pyautoencoder.variational.stochastic_layers import FullyFactorizedGaussian 

# ======================= FullyFactorizedGaussian =======================

def test_ffg_initial_state():
    latent_dim = 5
    head = FullyFactorizedGaussian(latent_dim=latent_dim)

    assert head.latent_dim == latent_dim
    assert head.built is False
    assert head._built is False
    assert head.mu is None
    assert head.log_var is None
    # in_features should not exist before build
    assert not hasattr(head, "in_features")


def test_ffg_build_with_valid_input_sets_layers_and_flag():
    latent_dim = 4
    B, F = 3, 7
    x = torch.randn(B, F, dtype=torch.float64)

    head = FullyFactorizedGaussian(latent_dim=latent_dim)
    head.build(x)

    assert head.built is True
    assert head._built is True
    assert head.in_features == F

    assert isinstance(head.mu, nn.Linear)
    assert isinstance(head.log_var, nn.Linear)

    # Check layer shapes
    assert head.mu.in_features == F
    assert head.mu.out_features == latent_dim
    assert head.log_var.in_features == F
    assert head.log_var.out_features == latent_dim

    # Check device & dtype of parameters match input
    for p in head.mu.parameters():
        assert p.device == x.device
        assert p.dtype == x.dtype
    for p in head.log_var.parameters():
        assert p.device == x.device
        assert p.dtype == x.dtype


def test_ffg_build_rejects_non_tensor_and_bad_shapes():
    head = FullyFactorizedGaussian(latent_dim=3)

    # Non-tensor
    with pytest.raises(TypeError):
        head.build([1.0, 2.0])  # type: ignore[arg-type]

    # 1D tensor
    with pytest.raises(ValueError):
        head.build(torch.randn(10))

    # 3D tensor
    with pytest.raises(ValueError):
        head.build(torch.randn(2, 3, 4))

    # Zero feature dimension
    with pytest.raises(ValueError):
        head.build(torch.randn(2, 0))


def test_ffg_build_can_be_called_twice_with_same_feature_dim():
    """Docstring says idempotent for same F: at least must be safe to call twice."""
    B, F = 2, 5
    latent_dim = 3
    x1 = torch.randn(B, F)
    x2 = torch.randn(B + 1, F)  # different batch size, same F

    head = FullyFactorizedGaussian(latent_dim=latent_dim)
    head.build(x1)
    # Calling build again with same feature dimension should not error
    head.build(x2)

    assert head.built is True
    assert head.in_features == F
    assert isinstance(head.mu, nn.Linear)
    assert head.mu.in_features == F

def test_ffg_forward_raises_if_not_built():
    head = FullyFactorizedGaussian(latent_dim=3)
    x = torch.randn(2, 5)

    with pytest.raises(RuntimeError, match="not built\\. Call `.build\\(x\\)` first"):
        head(x)

def test_ffg_training_forward_shapes_and_grad_and_sampling():
    B, F, Dz, S = 4, 6, 3, 5
    x = torch.randn(B, F)

    head = FullyFactorizedGaussian(latent_dim=Dz)
    head.build(x)

    head.train()
    torch.set_grad_enabled(True)

    z, mu, log_var = head(x, S=S)

    # Shape checks
    assert z.shape == (B, S, Dz)
    assert mu.shape == (B, Dz)
    assert log_var.shape == (B, Dz)

    # Grad checks: since layers are Linear, outputs should require grad in training
    assert z.requires_grad is True
    assert mu.requires_grad is True
    assert log_var.requires_grad is True

    # Basic sanity: std computed from log_var should be finite
    std = torch.exp(0.5 * log_var)
    assert torch.isfinite(std).all()


def test_ffg_training_sampling_uses_global_rng():
    """Two calls with the same seed should produce identical samples."""
    B, F, Dz, S = 3, 5, 2, 4
    x = torch.randn(B, F)

    head = FullyFactorizedGaussian(latent_dim=Dz)
    head.build(x)
    head.train()
    torch.set_grad_enabled(True)

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
    head.build(x)

    head.eval()
    torch.set_grad_enabled(True)  # even if global grad enabled, eval path is deterministic

    z, mu, log_var = head(x, S=S)

    assert z.shape == (B, S, Dz)
    assert mu.shape == (B, Dz)
    assert log_var.shape == (B, Dz)

    # In eval mode, z should be just expanded mu
    expected_z = mu.unsqueeze(1).expand(-1, S, -1)
    assert torch.allclose(z, expected_z)

    # Still part of a differentiable graph (no inference_mode here), so grad is allowed
    assert z.requires_grad is True
    assert mu.requires_grad is True
    assert log_var.requires_grad is True

def test_ffg_eval_forward_respects_default_S_equals_1():
    B, F, Dz = 2, 4, 3
    x = torch.randn(B, F)

    head = FullyFactorizedGaussian(latent_dim=Dz)
    head.build(x)
    head.eval()

    z, mu, log_var = head(x)  # S default = 1

    assert z.shape == (B, 1, Dz)
    expected_z = mu.unsqueeze(1)  # [B, 1, Dz]
    assert torch.allclose(z, expected_z)


def test_ffg_rejects_invalid_S_values():
    """Test that S parameter validation rejects 0 and negative values."""
    B, F, Dz = 2, 4, 3
    x = torch.randn(B, F)

    head = FullyFactorizedGaussian(latent_dim=Dz)
    head.build(x)
    head.train()

    # S = 0 should raise
    with pytest.raises(ValueError, match="S must be >= 1"):
        head(x, S=0)

    # S < 0 should raise
    with pytest.raises(ValueError, match="S must be >= 1"):
        head(x, S=-1)

    # S = -5 should raise
    with pytest.raises(ValueError, match="S must be >= 1"):
        head(x, S=-5)
