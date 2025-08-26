"""Base loss functions for autoencoders."""
import math
import torch
import torch.nn.functional as F
from typing import Union
from enum import Enum

class LikelihoodType(Enum):
    """Supported likelihood types for the decoder distribution p(x|z)."""
    GAUSSIAN = 'gaussian'
    BERNOULLI = 'bernoulli'

# Cache for log(2π) constants per (device, dtype)
_LOG2PI_CACHE = {}

def _get_log2pi(x: torch.Tensor) -> torch.Tensor:
    """Return log(2π) cached for the given device/dtype."""
    key = (x.device, x.dtype)
    if key not in _LOG2PI_CACHE:
        _LOG2PI_CACHE[key] = torch.tensor(2.0 * math.pi, device=x.device, dtype=x.dtype).log()
    return _LOG2PI_CACHE[key]

def log_likelihood(x: torch.Tensor, 
                  x_hat: torch.Tensor, 
                  likelihood: Union[str, LikelihoodType] = LikelihoodType.GAUSSIAN) -> torch.Tensor:
    """
    Computes elementwise log-likelihood log p(x|x_hat) under different likelihood assumptions.
    
    For continuous data:
        Gaussian (σ² = 1):
            log p(x|x_hat) = -0.5 * [ (x - x_hat)^2 + log(2π) ]
        Each dimension contributes independently. To obtain per-sample log-likelihoods,
        sum over feature dimensions.
    
    For discrete data:
        Bernoulli:
            log p(x|x_hat) = x * log σ(x_hat) + (1 - x) * log(1 - σ(x_hat)),
        where σ is the sigmoid function and x_hat are logits.
    
    Args:
        x (torch.Tensor): Ground truth tensor.
        x_hat (torch.Tensor): Reconstructed tensor. For Bernoulli, values are logits.
        likelihood (Union[str, LikelihoodType]): Choice of likelihood model. Defaults to Gaussian.
    
    Returns:
        torch.Tensor: Elementwise log-likelihood with the same shape as `x`.
                      For multi-dimensional inputs, reduce across feature dimensions
                      to obtain per-sample log-likelihoods.
    
    Notes:
        - Bernoulli case uses a numerically stable BCE implementation in log-space.
        - Gaussian case assumes fixed unit variance (σ²=1) and includes the normalization constant.
        - log(2π) is cached per (device, dtype) for efficiency.
    """
    if isinstance(likelihood, str):
        likelihood = LikelihoodType(likelihood.lower())
    
    if likelihood == LikelihoodType.BERNOULLI:
        return -F.binary_cross_entropy_with_logits(x_hat, x, reduction='none')
    
    elif likelihood == LikelihoodType.GAUSSIAN:
        squared_error = (x_hat - x) ** 2
        log_2pi = _get_log2pi(x)
        return -0.5 * (squared_error + log_2pi)
    
    else:
        raise ValueError(f"Unsupported likelihood: {likelihood}")
