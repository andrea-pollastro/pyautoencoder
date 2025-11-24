import torch
from typing import Union, NamedTuple

from .base import log_likelihood, LikelihoodType

class ELBOComponents(NamedTuple):
    """Named tuple containing the main components of the ELBO.

    Attributes
    ----------
    elbo : torch.Tensor
        Scalar tensor containing the mean ELBO over the batch (with gradients).
    log_likelihood : torch.Tensor
        Scalar tensor containing the mean reconstruction term
        :math:`\mathbb{E}_{q(z \mid x)}[\log p(x \mid z)]`.
    beta_kl_divergence : torch.Tensor
        Scalar tensor containing :math:`\beta` times the mean KL divergence
        :math:`\mathrm{KL}(q(z \mid x) \| p(z))` over the batch.
    """

    elbo: torch.Tensor                 # scalar: batch-mean ELBO (with grad)
    log_likelihood: torch.Tensor       # scalar: batch-mean E_q[log p(x|z)]
    beta_kl_divergence: torch.Tensor   # scalar: batch-mean beta * KL(q||p)

def kl_divergence_gaussian(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    r"""Compute the KL divergence :math:`\mathrm{KL}(q(z \mid x) \,\|\, p(z))`
    for diagonal Gaussian posteriors.

    The approximate posterior is assumed to be

    .. math::

        q(z \mid x) = \mathcal{N}(\mu, \operatorname{diag}(\sigma^2)), \qquad
        \sigma^2 = \exp(\log \sigma^2),

    and the prior is the standard normal

    .. math::

        p(z) = \mathcal{N}(0, I).

    The closed-form KL divergence for each sample is

    .. math::

        \mathrm{KL}(q \,\|\, p) =
            -\tfrac{1}{2} \sum_{d}
            \bigl(1 + \log \sigma_d^2 - \mu_d^2 - \sigma_d^2 \bigr).

    Parameters
    ----------
    mu : torch.Tensor
        Mean tensor of shape ``[B, D_z]``.
    log_var : torch.Tensor
        Log-variance tensor of shape ``[B, D_z]``.

    Returns
    -------
    torch.Tensor
        Per-sample KL divergences of shape ``[B]``. Reduction is performed
        over latent dimensions but **not** over the batch.
    """

    return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)

def compute_ELBO(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    likelihood: Union[str, LikelihoodType] = LikelihoodType.GAUSSIAN,
    beta: float = 1.0,
) -> ELBOComponents:
    r"""Compute the Evidence Lower Bound (ELBO) for a (beta-)Variational Autoencoder.

    This function implements the beta-VAE objective:

    .. math::

        \mathcal{L}(x; \beta)
            = \mathbb{E}_{q(z \mid x)}[\log p(x \mid z)]
            \;-\;
            \beta \, \mathrm{KL}(q(z \mid x) \,\|\, p(z)).

    The reconstruction term :math:`\log p(x \mid z)` is computed using
    :func:`vae.base.log_likelihood`, which supports both Gaussian and
    Bernoulli likelihoods.

    Monte Carlo estimation
    ----------------------
    If ``x_hat`` contains ``S`` Monte Carlo samples, the expectation
    \ :math:`\mathbb{E}_{q(z \mid x)}` is approximated by:

    .. math::

        \mathbb{E}_{q(z \mid x)}[\log p(x \mid z)]
            \approx \frac{1}{S} \sum_{s=1}^{S}
            \log p(x \mid z^{(s)}).

    Inputs and broadcasting
    -----------------------
    - If ``x_hat`` has shape ``[B, ...]``, it is interpreted as a single
    sample and is expanded to ``[B, 1, ...]``.
    - ``x`` is broadcast to match the sample dimension.

    Parameters
    ----------
    x : torch.Tensor
        Ground-truth inputs, shape ``[B, ...]``.
    x_hat : torch.Tensor
        Reconstructed samples. Either of shape ``[B, ...]`` (single sample)
        or ``[B, S, ...]`` where ``S`` is the number of samples.
    mu : torch.Tensor
        Mean of the approximate posterior ``q(z \mid x)``, shape ``[B, D_z]``.
    log_var : torch.Tensor
        Log-variance of ``q(z \mid x)``, shape ``[B, D_z]``.
    likelihood : Union[str, LikelihoodType], optional
        Likelihood model for the reconstruction term. Defaults to Gaussian.
    beta : float, optional
        Weight for the KL term (beta-VAE). ``beta = 1`` yields the standard VAE.

    Returns
    -------
    ELBOComponents
        Named tuple containing:
        
        * **elbo** – Mean ELBO over the batch.
        * **log_likelihood** – Mean reconstruction term
        :math:`\mathbb{E}_{q}[\log p(x \mid z)]`.
        * **beta_kl_divergence** – Mean :math:`\beta \, \mathrm{KL}(q \,\|\, p)`
        over the batch.

    Notes
    -----
    - The log-likelihood term includes the Gaussian normalization constant
    or numerically stable Bernoulli log-probabilities.
    - All returned values are **batch means** and maintain gradients.
    - ``x_hat`` is never detached; gradients flow through the decoder.
    """

    # Ensure a sample dimension S exists -> [B, S, ...]
    if x_hat.dim() == x.dim():
        x_hat = x_hat.unsqueeze(1)  # S = 1
    B, S = x_hat.size(0), x_hat.size(1)

    # Broadcast x to match x_hat's [B, S, ...] shape
    x_expanded = x.unsqueeze(1)  # [B, 1, ...]
    if x_expanded.shape != x_hat.shape:
        # expand_as is a view (no real data copy) when only singleton dims are expanded
        x_expanded = x_expanded.expand_as(x_hat)

    # log p(x|z): elementwise -> sum over features => [B, S]
    log_px_z = log_likelihood(x_expanded, x_hat, likelihood=likelihood)
    log_px_z = log_px_z.reshape(B, S, -1).sum(-1)

    # E_q[log p(x|z)] via Monte Carlo average across S: [B]
    E_log_px_z = log_px_z.mean(dim=1)

    # KL(q||p): [B]
    kl_q_p = kl_divergence_gaussian(mu, log_var)

    # ELBO per sample and batch means (retain grads)
    elbo_per_sample = E_log_px_z - beta * kl_q_p          # [B]
    elbo = elbo_per_sample.mean()                         # scalar
    E_log_px_z_mean = E_log_px_z.mean()                   # scalar
    beta_kl_q_p_mean = beta * kl_q_p.mean()               # scalar

    return ELBOComponents(
        elbo=elbo,
        log_likelihood=E_log_px_z_mean,
        beta_kl_divergence=beta_kl_q_p_mean,
    )
