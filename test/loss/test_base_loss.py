import pytest
import math
import torch
import torch.nn.functional as F

from pyautoencoder.loss.base import ( 
    log_likelihood,
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
