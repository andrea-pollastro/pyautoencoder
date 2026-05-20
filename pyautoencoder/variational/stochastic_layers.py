import torch
import torch.nn as nn

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
    """

    def __init__(self, latent_dim: int):
        """Construct a Gaussian posterior head.

        Parameters
        ----------
        latent_dim : int
            Dimensionality of the latent space ``z``.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.mu = nn.LazyLinear(out_features=latent_dim)
        self.log_var = nn.LazyLinear(out_features=latent_dim)

    def get_params(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute posterior parameters without drawing samples.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``[B, F]``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(mu, log_var)``, each of shape ``[B, latent_dim]``.
        """

        return self.mu(x), self.log_var(x)  # type: ignore

    def forward(self, x: torch.Tensor, S: int = 1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            ``(z, mu, log_var)``, where:

            * ``z`` – sampled or repeated latent codes, shape ``[B, S, latent_dim]``.
            * ``mu`` – mean of :math:`q(z \mid x)`, shape ``[B, latent_dim]``.
            * ``log_var`` – log-variance of :math:`q(z \mid x)`, shape ``[B, latent_dim]``.

        Raises
        ------
        ValueError
            If ``S < 1``.
        """

        if S < 1:
            raise ValueError("S must be >= 1.")

        mu = self.mu(x)             # [B, Dz]
        log_var = self.log_var(x)   # [B, Dz]

        if self.training:
            z = self._reparametrize(mu=mu, log_var=log_var, S=S) # [B, S, Dz]
        else:
            z = mu.unsqueeze(1).expand(-1, S, -1)               # [B, S, Dz]

        return z, mu, log_var
    
    def _reparametrize(self, mu: torch.Tensor, log_var: torch.Tensor, S: int = 1) -> torch.Tensor:
        r"""Draw ``S`` latent samples via the reparameterization trick.

        .. math::

            z^{(s)} = \mu + \sigma \odot \epsilon^{(s)},
            \qquad \epsilon^{(s)} \sim \mathcal{N}(0, I).

        Parameters
        ----------
        mu : torch.Tensor
            Mean of the posterior, shape ``[B, D_z]``.
        log_var : torch.Tensor
            Log-variance of the posterior, shape ``[B, D_z]``.
        S : int, optional
            Number of samples to draw. Defaults to ``1``.

        Returns
        -------
        torch.Tensor
            Sampled latent codes of shape ``[B, S, D_z]``.
        """
        std = torch.exp(0.5 * log_var)              # [B, Dz]
        mu_e  = mu.unsqueeze(1).expand(-1, S, -1)   # [B, S, Dz]
        std_e = std.unsqueeze(1).expand(-1, S, -1)  # [B, S, Dz]
        eps = torch.randn_like(std_e)
        z = mu_e + std_e * eps                      # [B, S, Dz]
        return z
    