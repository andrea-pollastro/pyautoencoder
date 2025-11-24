import math
import torch
import torch.nn.functional as F
from typing import Union
from enum import Enum

class LikelihoodType(Enum):
    """Enumeration of supported decoder likelihood models :math:`p(x \mid z)`.

    Values
    ------
    GAUSSIAN : str
        Gaussian likelihood with fixed unit variance :math:`\sigma^2 = 1`.
    BERNOULLI : str
        Bernoulli likelihood for discrete data, with ``x_hat`` interpreted
        as logits.
    """

    GAUSSIAN = 'gaussian'
    BERNOULLI = 'bernoulli'

# Cache for log(2pi) constants per (device, dtype)
_LOG2PI_CACHE = {}

def _get_log2pi(x: torch.Tensor) -> torch.Tensor:
    """Return a cached value of :math:`\log(2\pi)` for the given device and dtype.

    This avoids repeatedly allocating the constant for different devices or
    precisions. A separate tensor is cached for each ``(device, dtype)`` pair.

    Parameters
    ----------
    x : torch.Tensor
        A tensor whose ``device`` and ``dtype`` determine which cached value is
        returned or created.

    Returns
    -------
    torch.Tensor
        A scalar tensor equal to :math:`\log(2\pi)` with the same device and
        dtype as ``x``.
    """

    key = (x.device, x.dtype)
    if key not in _LOG2PI_CACHE:
        _LOG2PI_CACHE[key] = torch.tensor(2.0 * math.pi, device=x.device, dtype=x.dtype).log()
    return _LOG2PI_CACHE[key]

def log_likelihood(x: torch.Tensor, 
                   x_hat: torch.Tensor, 
                   likelihood: Union[str, LikelihoodType] = LikelihoodType.GAUSSIAN) -> torch.Tensor:
    r"""Compute the elementwise log-likelihood :math:`\log p(x \mid \hat{x})`.

    Two likelihood models are supported.

    - Gaussian (continuous data)
      Assuming fixed unit variance :math:`\sigma^2 = 1`, each element follows:

      .. math::

          \log p(x \mid \hat{x}) =
              -\tfrac{1}{2} \left[ (x - \hat{x})^2 + \log(2\pi) \right].

      The output has the same shape as ``x``. Summing over feature dimensions
      gives per-sample log-likelihoods.

    - Bernoulli (discrete data)
      Here ``x_hat`` is interpreted as logits. Each element follows:

      .. math::

          \log p(x \mid \hat{x}) =
              x \log \sigma(\hat{x})
              + (1 - x) \log\!\left( 1 - \sigma(\hat{x}) \right),

      where :math:`\sigma` is the sigmoid. A numerically stable implementation
      using :func:`torch.nn.functional.binary_cross_entropy_with_logits`
      is applied.

    Parameters
    ----------
    x : torch.Tensor
        Ground-truth tensor.
    x_hat : torch.Tensor
        Reconstructed tensor. For the Bernoulli case, values are logits.
    likelihood : Union[str, LikelihoodType], optional
        Likelihood model to use. May be a string (``"gaussian"``,
        ``"bernoulli"``) or a :class:`LikelihoodType` enum value.
        Defaults to Gaussian.

    Returns
    -------
    torch.Tensor
        Elementwise log-likelihood with the same shape as ``x``.

    Notes
    -----
    - The Gaussian case includes the normalization constant
      :math:`\log(2\pi)`, cached per ``(device, dtype)`` with
      :func:`_get_log2pi`.
    - The Bernoulli case is fully numerically stable because it operates
      directly in log-space.
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
