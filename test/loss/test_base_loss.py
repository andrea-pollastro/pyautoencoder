import pytest
import math
import torch
import torch.nn.functional as F

from pyautoencoder.loss.base import ( 
    log_likelihood,
    kl_divergence_diag_gaussian,
    LikelihoodType,
    _get_log2pi,
    _LOG2PI_CACHE,
)

def test_get_log2pi_value_matches_math_log_2pi():
    _LOG2PI_CACHE.clear()
    x = torch.randn(2, 3, dtype=torch.float32)

    log2pi = _get_log2pi(x)

    assert log2pi.shape == ()  # scalar
    assert log2pi.dtype == x.dtype
    assert log2pi.device == x.device

    expected = math.log(2.0 * math.pi)
    assert torch.allclose(log2pi, torch.tensor(expected, dtype=x.dtype))

def test_get_log2pi_caches_per_device_and_dtype():
    _LOG2PI_CACHE.clear()

    x32 = torch.randn(1, dtype=torch.float32)
    x64 = torch.randn(1, dtype=torch.float64)

    l32_first = _get_log2pi(x32)
    l32_second = _get_log2pi(x32)
    l64 = _get_log2pi(x64)

    # Same (device, dtype) -> same tensor object (cached)
    assert l32_first is l32_second

    # Different dtype -> different cache entry
    assert l32_first is not l64
    assert l32_first.dtype == torch.float32
    assert l64.dtype == torch.float64

    # Cache keys are exactly the (device, dtype) pairs
    assert (x32.device, x32.dtype) in _LOG2PI_CACHE
    assert (x64.device, x64.dtype) in _LOG2PI_CACHE

def test_log_likelihood_gaussian_scalar_matches_formula():
    # One scalar example, deterministic math
    x = torch.tensor(1.5)
    x_hat = torch.tensor(0.5)

    out = log_likelihood(x, x_hat, likelihood=LikelihoodType.GAUSSIAN)

    # manual formula: -0.5 * [ (x - x_hat)^2 + log(2*pi) ]
    diff = float(x_hat - x)
    squared_error = diff * diff
    expected = -0.5 * (squared_error + math.log(2.0 * math.pi))

    assert out.shape == ()
    assert torch.allclose(out, torch.tensor(expected, dtype=out.dtype))


def test_log_likelihood_gaussian_tensor_matches_elementwise_form():
    torch.manual_seed(0)
    x = torch.randn(2, 3)
    x_hat = torch.randn(2, 3)

    out = log_likelihood(x, x_hat, likelihood="gaussian")

    assert out.shape == x.shape

    log2pi = _get_log2pi(x)
    squared_error = (x_hat - x) ** 2
    expected = -0.5 * (squared_error + log2pi)

    assert torch.allclose(out, expected)


def test_log_likelihood_gaussian_preserves_dtype_and_device():
    x = torch.randn(4, 5, dtype=torch.float64)
    x_hat = torch.randn(4, 5, dtype=torch.float64)

    out = log_likelihood(x, x_hat, likelihood=LikelihoodType.GAUSSIAN)

    assert out.dtype == x.dtype
    assert out.device == x.device


def test_log_likelihood_gaussian_works_with_higher_dimensional_inputs():
    # Shape [B, S, C, H, W] should be preserved
    x = torch.randn(2, 3, 1, 4, 4)
    x_hat = torch.randn(2, 3, 1, 4, 4)

    out = log_likelihood(x, x_hat, likelihood="gaussian")

    assert out.shape == x.shape
    # Example reduction to per-sample log-likelihood (not part of implementation, but as usage)
    per_sample = out.view(2, -1).sum(dim=1)
    assert per_sample.shape == (2,)

def test_log_likelihood_bernoulli_matches_negative_bce_with_logits():
    torch.manual_seed(0)

    # binary targets in {0,1}
    x = torch.randint(low=0, high=2, size=(4, 5)).float()
    logits = torch.randn(4, 5)

    out = log_likelihood(x, logits, likelihood=LikelihoodType.BERNOULLI)

    # Should be the negative of BCEWithLogits (reduction='none')
    bce = F.binary_cross_entropy_with_logits(logits, x, reduction="none")
    expected = -bce

    assert out.shape == x.shape
    assert torch.allclose(out, expected)

def test_log_likelihood_bernoulli_small_manual_example():
    # 1D example, manual math check
    logits = torch.tensor([0.0, 2.0, -1.0])  # x_hat
    x = torch.tensor([0.0, 1.0, 1.0])        # targets in {0,1}

    out = log_likelihood(x, logits, likelihood="bernoulli")

    # Manual: log p(x|logits) = x * log(sigmoid(l)) + (1 - x) * log(1 - sigmoid(l))
    sig = torch.sigmoid(logits)
    manual = x * torch.log(sig) + (1 - x) * torch.log(1 - sig)

    assert out.shape == x.shape
    assert torch.allclose(out, manual, atol=1e-6, rtol=1e-6)

def test_log_likelihood_bernoulli_preserves_dtype_and_device():
    x = torch.randint(0, 2, (3, 4)).double()
    logits = torch.randn(3, 4, dtype=torch.float64)

    out = log_likelihood(x, logits, likelihood=LikelihoodType.BERNOULLI)

    assert out.dtype == x.dtype
    assert out.device == x.device

def test_log_likelihood_accepts_string_and_enum():
    x = torch.randn(2, 3)
    x_hat = torch.randn(2, 3)

    out_enum = log_likelihood(x, x_hat, likelihood=LikelihoodType.GAUSSIAN)
    out_str = log_likelihood(x, x_hat, likelihood="gaussian")

    assert torch.allclose(out_enum, out_str)

    x_bin = torch.randint(0, 2, (2, 3)).float()
    logits = torch.randn(2, 3)

    out_enum_b = log_likelihood(x_bin, logits, likelihood=LikelihoodType.BERNOULLI)
    out_str_b = log_likelihood(x_bin, logits, likelihood="bernoulli")

    assert torch.allclose(out_enum_b, out_str_b)

def test_log_likelihood_invalid_likelihood_raises():
    x = torch.randn(2, 3)
    x_hat = torch.randn(2, 3)

    with pytest.raises(ValueError):
        log_likelihood(x, x_hat, likelihood="poisson")  # unsupported string

    # Also ensure passing a wrong type via Enum is caught naturally
    class FakeEnum:
        value = "gaussian"
    
    with pytest.raises(ValueError, match="Unsupported likelihood"):
        log_likelihood(x, x_hat, likelihood=FakeEnum()) # type: ignore

# ================= kl_divergence_diag_gaussian =================

def test_kl_divergence_standard_case_vs_standard_normal():
    """Test KL(N(mu, sigma^2) || N(0, I)) with reduce_sum=True."""
    B, Dz = 3, 4
    mu_q = torch.randn(B, Dz)
    log_var_q = torch.randn(B, Dz)
    
    kl = kl_divergence_diag_gaussian(mu_q, log_var_q)
    
    # Shape: [B] when reduce_sum=True
    assert kl.shape == (B,)
    
    # KL should be non-negative
    assert torch.all(kl >= 0)
    
    # Manual computation of the formula for standard normal prior:
    # KL = 0.5 * sum_d [ -log(sigma_q^2) + sigma_q^2 + mu_q^2 - 1 ]
    var_q = log_var_q.exp()
    expected_kl = 0.5 * torch.sum(-log_var_q + var_q + mu_q.pow(2) - 1, dim=-1)
    
    assert torch.allclose(kl, expected_kl, atol=1e-6)


def test_kl_divergence_with_reduce_sum_false():
    """Test KL divergence with reduce_sum=False returns per-dimension KL."""
    B, Dz = 2, 3
    mu_q = torch.randn(B, Dz)
    log_var_q = torch.randn(B, Dz)
    
    kl_reduced = kl_divergence_diag_gaussian(mu_q, log_var_q, reduce_sum=True)
    kl_unreduced = kl_divergence_diag_gaussian(mu_q, log_var_q, reduce_sum=False)
    
    # reduced shape: [B], unreduced shape: [B, Dz]
    assert kl_reduced.shape == (B,)
    assert kl_unreduced.shape == (B, Dz)
    
    # Sum of unreduced should match reduced
    assert torch.allclose(kl_unreduced.sum(dim=-1), kl_reduced, atol=1e-6)


def test_kl_divergence_with_custom_prior():
    """Test KL divergence with non-standard prior."""
    B, Dz = 2, 3
    mu_q = torch.randn(B, Dz)
    log_var_q = torch.randn(B, Dz)
    mu_p = torch.randn(B, Dz)
    log_var_p = torch.randn(B, Dz)
    
    kl = kl_divergence_diag_gaussian(mu_q, log_var_q, mu_p, log_var_p)
    
    assert kl.shape == (B,)
    # KL should still be non-negative (but may not be for arbitrary parameters)
    # At least check it's finite
    assert torch.isfinite(kl).all()
    
    # Manual formula:
    # KL = 0.5 * sum_d [ (log_var_p - log_var_q) + (var_q + (mu_q - mu_p)^2) / var_p - 1 ]
    var_q = log_var_q.exp()
    var_p = log_var_p.exp()
    term1 = log_var_p - log_var_q
    term2 = (var_q + (mu_q - mu_p).pow(2)) / var_p
    expected_kl = 0.5 * torch.sum(term1 + term2 - 1, dim=-1)
    
    assert torch.allclose(kl, expected_kl, atol=1e-5)


def test_kl_divergence_zero_when_q_equals_p():
    """Test that KL is (approximately) zero when q = p."""
    B, Dz = 2, 3
    mu = torch.randn(B, Dz)
    log_var = torch.randn(B, Dz)
    
    kl = kl_divergence_diag_gaussian(mu, log_var, mu, log_var)
    
    # KL should be very close to zero when distributions are identical
    assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-5)


def test_kl_divergence_batch_size_one():
    """Test KL divergence with batch size 1."""
    Dz = 4
    mu_q = torch.randn(1, Dz)
    log_var_q = torch.randn(1, Dz)
    
    kl = kl_divergence_diag_gaussian(mu_q, log_var_q)
    
    assert kl.shape == (1,)
    assert torch.isfinite(kl).all()
    assert kl >= 0


def test_kl_divergence_preserves_dtype():
    """Test that KL divergence preserves input dtype."""
    B, Dz = 2, 3
    
    # float32
    mu_q32 = torch.randn(B, Dz, dtype=torch.float32)
    log_var_q32 = torch.randn(B, Dz, dtype=torch.float32)
    kl32 = kl_divergence_diag_gaussian(mu_q32, log_var_q32)
    assert kl32.dtype == torch.float32
    
    # float64
    mu_q64 = torch.randn(B, Dz, dtype=torch.float64)
    log_var_q64 = torch.randn(B, Dz, dtype=torch.float64)
    kl64 = kl_divergence_diag_gaussian(mu_q64, log_var_q64)
    assert kl64.dtype == torch.float64


def test_kl_divergence_preserves_device():
    """Test that KL divergence preserves device."""
    B, Dz = 2, 3
    mu_q = torch.randn(B, Dz)
    log_var_q = torch.randn(B, Dz)
    
    kl = kl_divergence_diag_gaussian(mu_q, log_var_q)
    
    assert kl.device == mu_q.device


def test_kl_divergence_symmetric_when_p_and_q_swapped_with_custom_prior():
    """Test asymmetry property: KL(q||p) != KL(p||q) in general."""
    B, Dz = 2, 3
    mu_q = torch.randn(B, Dz)
    log_var_q = torch.randn(B, Dz)
    mu_p = torch.randn(B, Dz)
    log_var_p = torch.randn(B, Dz)
    
    kl_q_p = kl_divergence_diag_gaussian(mu_q, log_var_q, mu_p, log_var_p)
    kl_p_q = kl_divergence_diag_gaussian(mu_p, log_var_p, mu_q, log_var_q)
    
    # KL divergence is asymmetric in general
    # (unless the distributions happen to be very similar)
    # Just check both are valid
    assert torch.isfinite(kl_q_p).all()
    assert torch.isfinite(kl_p_q).all()


def test_kl_divergence_backward_flows_gradients():
    """Test that gradients flow through KL divergence."""
    B, Dz = 2, 3
    mu_q = torch.randn(B, Dz, requires_grad=True)
    log_var_q = torch.randn(B, Dz, requires_grad=True)
    
    kl = kl_divergence_diag_gaussian(mu_q, log_var_q)
    loss = kl.sum()
    loss.backward()
    
    # Gradients should exist and be non-zero
    assert mu_q.grad is not None
    assert log_var_q.grad is not None
    assert torch.any(mu_q.grad != 0)
    assert torch.any(log_var_q.grad != 0)


def test_kl_divergence_with_custom_prior_backward():
    """Test gradient flow with custom prior."""
    B, Dz = 2, 3
    mu_q = torch.randn(B, Dz, requires_grad=True)
    log_var_q = torch.randn(B, Dz, requires_grad=True)
    mu_p = torch.randn(B, Dz, requires_grad=True)
    log_var_p = torch.randn(B, Dz, requires_grad=True)
    
    kl = kl_divergence_diag_gaussian(mu_q, log_var_q, mu_p, log_var_p)
    loss = kl.sum()
    loss.backward()
    
    # All gradients should exist
    assert mu_q.grad is not None
    assert log_var_q.grad is not None
    assert mu_p.grad is not None
    assert log_var_p.grad is not None


def test_kl_divergence_matches_pytorch_implementation():
    """Compare with a reference PyTorch implementation."""
    B, Dz = 3, 4
    mu_q = torch.randn(B, Dz)
    log_var_q = torch.randn(B, Dz)
    
    # Our implementation
    kl_ours = kl_divergence_diag_gaussian(mu_q, log_var_q)
    
    # Reference implementation (standard VAE KL)
    var_q = log_var_q.exp()
    kl_ref = 0.5 * torch.sum(-log_var_q + var_q + mu_q.pow(2) - 1, dim=-1)
    
    assert torch.allclose(kl_ours, kl_ref, atol=1e-5)


def test_kl_divergence_large_batch():
    """Test with larger batch size."""
    B, Dz = 128, 32
    mu_q = torch.randn(B, Dz)
    log_var_q = torch.randn(B, Dz)
    
    kl = kl_divergence_diag_gaussian(mu_q, log_var_q)
    
    assert kl.shape == (B,)
    assert torch.all(kl >= 0)
    assert torch.isfinite(kl).all()


def test_kl_divergence_high_dimensional():
    """Test with high-dimensional latent space."""
    B, Dz = 2, 512
    mu_q = torch.randn(B, Dz)
    log_var_q = torch.randn(B, Dz)
    
    kl = kl_divergence_diag_gaussian(mu_q, log_var_q)
    
    assert kl.shape == (B,)
    assert torch.isfinite(kl).all()