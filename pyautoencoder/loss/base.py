import math
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Union, Optional, Dict
from enum import Enum

class LikelihoodType(Enum):
    r"""Enumeration of supported decoder likelihood models :math:`p(x \mid z)`.

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

@dataclass(slots=True, repr=True)
class LossResult:
    r"""Container for loss computation results with objective and diagnostics.

    This dataclass holds the output of model loss computation methods
    (:meth:`AE.compute_loss`, :meth:`VAE.compute_loss`, etc.), separating
    the optimizable objective from optional diagnostic metrics.

    Attributes
    ----------
    objective : torch.Tensor
        Scalar loss to optimize (e.g., negative log-likelihood or negative ELBO).
        Maintains gradient information for backpropagation.
    diagnostics : Dict[str, float]
        Dictionary of scalar metrics for monitoring and logging.
        Values are detached float scalars (not tensors) and do not track gradients.
        Examples include per-dimension NLL, KL divergence, ELBO, MSE, etc.
    """

    objective: torch.Tensor
    diagnostics: Dict[str, float]

def _get_log2pi(x: torch.Tensor) -> torch.Tensor:
    r"""Return a cached value of :math:`\log(2\pi)` for the given device and dtype.

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
    
def kl_divergence_diag_gaussian(
    mu_q: torch.Tensor, 
    log_var_q: torch.Tensor, 
    mu_p: Optional[torch.Tensor] = None, 
    log_var_p: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
    r"""Compute the KL divergence :math:`\mathrm{KL}(q(z \mid x) \,\|\, p(z))`
    between two diagonal Gaussian distributions.

    The approximate posterior is 
    :math:`q(z \mid x) = \mathcal{N}(\mu_q, \operatorname{diag}(\exp(\log \sigma_q^2)))`.
    
    The prior is 
    :math:`p(z) = \mathcal{N}(\mu_p, \operatorname{diag}(\exp(\log \sigma_p^2)))`.
    If :math:`\mu_p` and :math:`\log \sigma_p^2` are None, :math:`p(z) = \mathcal{N}(0, I)`.

    The closed-form KL divergence is:

    .. math::

        \mathrm{KL}(q \,\|\, p) = \frac{1}{2} \sum_{d} \left( 
            (\log \sigma_{p,d}^2 - \log \sigma_{q,d}^2) + 
            \frac{\exp(\log \sigma_{q,d}^2) + (\mu_{q,d} - \mu_{p,d})^2}{\exp(\log \sigma_{p,d}^2)} - 1 
        \right)

    Parameters
    ----------
    mu_q : torch.Tensor
        Mean of the first distribution ``[B, D_z]``.
    log_var_q : torch.Tensor
        Log-variance of the first distribution ``[B, D_z]``.
    mu_p : torch.Tensor, optional
        Mean of the second distribution ``[B, D_z]``. Defaults to 0.
    log_var_p : torch.Tensor, optional
        Log-variance of the second distribution ``[B, D_z]``. Defaults to 0.

    Returns
    -------
    torch.Tensor
        Per-sample KL divergences of shape [B].
    """
    
    # Se p non è specificata, assumiamo N(0, I)
    if mu_p is None:
        mu_p = torch.zeros_like(mu_q)
    if log_var_p is None:
        log_var_p = torch.zeros_like(log_var_q)

    # Calcolo dei termini
    var_q = log_var_q.exp()
    var_p = log_var_p.exp()
    
    term1 = log_var_p - log_var_q
    term2 = (var_q + (mu_q - mu_p).pow(2)) / var_p
    
    return 0.5 * torch.sum(term1 + term2 - 1, dim=-1)
