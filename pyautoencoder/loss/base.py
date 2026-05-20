import torch
import torch.nn.functional as F
from dataclasses import dataclass
from enum import Enum

class LikelihoodType(Enum):
    r"""Enumeration of supported decoder likelihood models :math:`p(x \mid z)`.

    Attributes
    ----------
    GAUSSIAN : str
        Gaussian likelihood with fixed unit variance :math:`\sigma^2 = 1`.
    BERNOULLI : str
        Bernoulli likelihood for discrete data, with ``x_hat`` interpreted
        as logits.
    """

    GAUSSIAN = 'gaussian'
    BERNOULLI = 'bernoulli'

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
    diagnostics : dict[str, float]
        Dictionary of scalar metrics for monitoring and logging.
        Values are detached float scalars (not tensors) and do not track gradients.
        Examples include log-likelihood, KL divergence, and ELBO.
    """

    objective: torch.Tensor
    diagnostics: dict[str, float]

def log_likelihood(x: torch.Tensor, 
                   x_hat: torch.Tensor, 
                   likelihood: str | LikelihoodType = LikelihoodType.GAUSSIAN) -> torch.Tensor:
    r"""Compute the elementwise log-likelihood :math:`\log p(x \mid \hat{x})`.

    Two likelihood models are supported.

    - Gaussian (continuous data)
      Assuming fixed unit variance :math:`\sigma^2 = 1`, each element follows:

      .. math::

          \log p(x \mid \hat{x}) =
              -\tfrac{1}{2} (x - \hat{x})^2.

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
        Ground-truth tensor of shape ``[B, ...]``.
    x_hat : torch.Tensor
        Reconstructed tensor of shape ``[B, ...]``. For the Bernoulli case,
        values are logits.
    likelihood : str | LikelihoodType, optional
        Likelihood model to use. May be a string (``"gaussian"``,
        ``"bernoulli"``) or a :class:`LikelihoodType` enum value.
        Defaults to Gaussian.

    Returns
    -------
    torch.Tensor
        Elementwise log-likelihood with the same shape as ``x``.

    Notes
    -----
    - The Gaussian case omits the normalization constant
      :math:`-\tfrac{1}{2}\log(2\pi)`, which is constant with respect to
      the model parameters and has no effect on optimization.
    - The Bernoulli case is fully numerically stable because it operates
      directly in log-space.
    """

    if isinstance(likelihood, str):
        likelihood = LikelihoodType(likelihood.lower())
    
    if likelihood == LikelihoodType.BERNOULLI:
        return -F.binary_cross_entropy_with_logits(x_hat, x, reduction='none')
    
    elif likelihood == LikelihoodType.GAUSSIAN:
        squared_error = (x_hat - x) ** 2
        return -0.5 * squared_error
    
    else:
        raise ValueError(f"Unsupported likelihood: {likelihood}")
    
def kl_divergence_diag_gaussian(
    mu_q: torch.Tensor, 
    log_var_q: torch.Tensor, 
    mu_p: torch.Tensor | None = None,
    log_var_p: torch.Tensor | None = None,
    reduce_sum: bool = True) -> torch.Tensor:
    r"""Compute the KL divergence :math:`\mathrm{KL}(q \,\|\, p)` between two
    diagonal Gaussian distributions.

    The first distribution is
    :math:`q = \mathcal{N}(\mu_q, \operatorname{diag}(\exp(\log \sigma_q^2)))`.

    The second distribution is
    :math:`p = \mathcal{N}(\mu_p, \operatorname{diag}(\exp(\log \sigma_p^2)))`.
    When :math:`\mu_p` and :math:`\log \sigma_p^2` are ``None``,
    :math:`p = \mathcal{N}(0, I)`.

    The closed-form KL divergence is:

    .. math::

        \mathrm{KL}(q \,\|\, p) = \frac{1}{2} \sum_{d} \left( 
            (\log \sigma_{p,d}^2 - \log \sigma_{q,d}^2) + 
            \frac{\exp(\log \sigma_{q,d}^2) + (\mu_{q,d} - \mu_{p,d})^2}{\exp(\log \sigma_{p,d}^2)} - 1 
        \right)

    Parameters
    ----------
    mu_q : torch.Tensor
        Mean of the first distribution, shape ``[B, D_z]``.
    log_var_q : torch.Tensor
        Log-variance of the first distribution, shape ``[B, D_z]``.
    mu_p : torch.Tensor or None, optional
        Mean of the second distribution, shape ``[B, D_z]``. Defaults to
        ``None``, which is treated as **0** (standard normal mean).
    log_var_p : torch.Tensor or None, optional
        Log-variance of the second distribution, shape ``[B, D_z]``. Defaults
        to ``None``, which is treated as **0** (standard normal log-variance).
    reduce_sum : bool, optional
        Sum over the dimensions. Defaults to ``True``.

    Returns
    -------
    torch.Tensor
        KL divergences of shape ``[B]`` when ``reduce_sum=True``, or
        ``[B, D_z]`` when ``reduce_sum=False``.
    """
    
    if mu_p is None and log_var_p is None:
        # Closed form for KL(N(mu_q, exp(log_var_q)) || N(0, I)) — avoids allocating zero tensors.
        kl = 0.5 * (log_var_q.exp() + mu_q.pow(2) - 1 - log_var_q)
        return kl.sum(dim=-1) if reduce_sum else kl

    if mu_p is None:
        mu_p = torch.zeros_like(mu_q)
    if log_var_p is None:
        log_var_p = torch.zeros_like(log_var_q)

    var_q = log_var_q.exp()
    var_p = log_var_p.exp()

    term1 = log_var_p - log_var_q
    term2 = (var_q + (mu_q - mu_p).pow(2)) / var_p
    kl = 0.5 * (term1 + term2 - 1)
    return kl.sum(dim=-1) if reduce_sum else kl
