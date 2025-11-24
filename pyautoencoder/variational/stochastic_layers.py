import torch
import torch.nn as nn
from typing import Optional
from .._base.base import NotBuiltError

class FullyFactorizedGaussian(nn.Module):
    r"""Gaussian posterior head producing a fully factorized :math:`q(z \mid x)`.

    Given input features ``x`` of shape ``[B, F]``, this module produces the
    parameters of a diagonal Gaussian posterior,

    .. math::

        q(z \mid x) = \mathcal{N}(z \mid \mu(x), \operatorname{diag}(\sigma(x)^2)),

    and (optionally) samples ``S`` latent draws via the reparameterization
    trick during training.

    The build step infers ``F`` and lazily constructs the linear layers
    ``mu`` and ``log_var``.

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the latent space ``z``.
    """

    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.mu: Optional[nn.Linear] = None
        self.log_var: Optional[nn.Linear] = None
        self._built = False

    def build(self, input_sample: torch.Tensor) -> None:
        """Initialize the linear layers using a representative input sample.

        This method infers the input feature dimension ``F`` from a tensor of
        shape ``[B, F]`` and constructs:

        * ``mu`` – a linear layer mapping ``F → latent_dim``.
        * ``log_var`` – a linear layer mapping ``F → latent_dim``.

        The method is idempotent as long as subsequent inputs have the same
        feature dimension.

        Parameters
        ----------
        input_sample : torch.Tensor
            Tensor of shape ``[B, F]`` used to infer ``F`` and initialize the
            corresponding linear layers.

        Raises
        ------
        TypeError
            If ``input_sample`` is not a tensor.
        ValueError
            If the tensor does not have shape ``[B, F]`` or if ``F <= 0``.
        """

        if not isinstance(input_sample, torch.Tensor):
            raise TypeError("build(x) expects a torch.Tensor.")
        if input_sample.ndim != 2:
            raise ValueError(f"build(x): expected shape [B, F], got {tuple(input_sample.shape)}. Flatten upstream.")
        if input_sample.shape[1] <= 0:
            raise ValueError("build(x): F (feature dimension) must be > 0.")
        
        in_features = int(input_sample.shape[1])

        self.mu = nn.Linear(in_features, self.latent_dim)
        self.log_var = nn.Linear(in_features, self.latent_dim)
        self.to(device=input_sample.device, dtype=input_sample.dtype)

        self.in_features = in_features
        self._built = True

    def forward(self, x: torch.Tensor, S: int = 1):
        r"""Compute parameters and (optionally) samples from the Gaussian posterior.

        During training, this method returns ``S`` Monte Carlo samples using the
        reparameterization trick:

        .. math::

            z^{(s)} = \mu + \sigma \odot \epsilon^{(s)},
            \qquad \epsilon^{(s)} \sim \mathcal{N}(0, I).

        During evaluation (``model.eval()``), deterministic output is returned
        with ``z`` equal to the repeated mean.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``[B, F]``.
        S : int, optional
            Number of Monte Carlo samples to generate. Must be ``>= 1``.
            Defaults to ``1`` (single sample).

        Returns
        -------
        tuple
            ``(z, mu, log_var)``, where:

            * ``z`` – sampled or repeated latent codes, shape ``[B, S, latent_dim]``.
            * ``mu`` – mean of :math:`q(z \mid x)`, shape ``[B, latent_dim]``.
            * ``log_var`` – log-variance of :math:`q(z \mid x)`, shape ``[B, latent_dim]``.

        Raises
        ------
        NotBuiltError
            If the module has not been built.
        ValueError
            If ``S < 1``.
        """

        if not self._built:
            raise NotBuiltError("FullyFactorizedGaussian not built. Call `.build(x)` first.")
        if S < 1:
            raise ValueError("S must be >= 1.")

        mu = self.mu(x)            # type: ignore       # [B, Dz]
        log_var = self.log_var(x)  # type: ignore       # [B, Dz]

        if self.training:
            std = torch.exp(0.5 * log_var)              # [B, Dz]
            mu_e  = mu.unsqueeze(1).expand(-1, S, -1)   # [B, S, Dz]
            std_e = std.unsqueeze(1).expand(-1, S, -1)  # [B, S, Dz]
            eps = torch.randn_like(std_e)
            z = mu_e + std_e * eps                      # [B, S, Dz]
        else:
            z = mu.unsqueeze(1).expand(-1, S, -1)       # [B, S, Dz]

        return z, mu, log_var
    
    @property
    def built(self) -> bool:
        """Whether the module has been successfully built.

        Returns
        -------
        bool
            ``True`` if :meth:`build` has been called and the internal layers have
            been initialized, ``False`` otherwise.
        """

        return self._built