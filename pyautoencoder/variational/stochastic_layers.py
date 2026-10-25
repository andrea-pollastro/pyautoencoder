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

        Raises
        ------
        ValueError
            If ``x`` is not of shape ``[B, F]``.
        """

        self._check_features(x)
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
            If ``S < 1`` or if ``x`` is not of shape ``[B, F]``.
        """

        _check_num_samples(S)
        mu, log_var = self.get_params(x)  # [B, Dz] each

        if self.training:
            z = self.reparametrize(mu=mu, log_var=log_var, S=S)  # [B, S, Dz]
        else:
            z = mu.unsqueeze(1).expand(-1, S, -1)               # [B, S, Dz]

        return z, mu, log_var

    def reparametrize(self, mu: torch.Tensor, log_var: torch.Tensor, S: int = 1) -> torch.Tensor:
        r"""Draw ``S`` latent samples via the reparameterization trick.

        .. math::

            z^{(s)} = \mu + \sigma \odot \epsilon^{(s)},
            \qquad \epsilon^{(s)} \sim \mathcal{N}(0, I).

        Unlike :meth:`forward`, this method always samples, regardless of
        train/eval mode. It is differentiable with respect to ``mu`` and
        ``log_var``, so it can be used on posterior parameters computed
        elsewhere (for example the adapted posteriors of :class:`AdaGVAE`).

        Parameters
        ----------
        mu : torch.Tensor
            Mean of the posterior, shape ``[B, D_z]``.
        log_var : torch.Tensor
            Log-variance of the posterior, shape ``[B, D_z]``.
        S : int, optional
            Number of samples to draw. Must be ``>= 1``. Defaults to ``1``.

        Returns
        -------
        torch.Tensor
            Sampled latent codes of shape ``[B, S, D_z]``.

        Raises
        ------
        ValueError
            If ``S < 1``, or if ``mu`` and ``log_var`` are not both of
            shape ``[B, D_z]``.
        """
        _check_num_samples(S)
        if mu.dim() != 2 or mu.shape != log_var.shape:
            raise ValueError(
                "mu and log_var must both have shape [B, D_z], got "
                f"{tuple(mu.shape)} and {tuple(log_var.shape)}."
            )

        std = torch.exp(0.5 * log_var)              # [B, Dz]
        mu_e  = mu.unsqueeze(1).expand(-1, S, -1)   # [B, S, Dz]
        std_e = std.unsqueeze(1).expand(-1, S, -1)  # [B, S, Dz]
        eps = torch.randn_like(std_e)
        z = mu_e + std_e * eps                      # [B, S, Dz]
        return z

    @staticmethod
    def _check_features(x: torch.Tensor) -> None:
        if x.dim() != 2:
            raise ValueError(
                "FullyFactorizedGaussian expects encoder features of shape [B, F], "
                f"got {tuple(x.shape)}. The encoder must output one flat feature "
                "vector per sample, e.g. by ending it with nn.Flatten()."
            )


def _check_num_samples(S: int) -> None:
    if S < 1:
        raise ValueError(f"S must be >= 1, got {S}.")
