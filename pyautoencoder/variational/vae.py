import torch
import torch.nn as nn
from dataclasses import dataclass
from ..loss.base import (
    LikelihoodType,
    log_likelihood,
    kl_divergence_diag_gaussian,
    LossResult
)
from .._base.base import BaseAutoencoder, ModelOutput
# Import the module, not the class: FullyFactorizedGaussian must be importable
# only from pyautoencoder.variational.stochastic_layers, not from this module.
from . import stochastic_layers

@dataclass(slots=True, repr=False)
class VAEEncodeOutput(ModelOutput):
    r"""Output of the VAE encoder stage.

    Attributes
    ----------
    z : torch.Tensor
        Latent codes of shape ``[B, S, D_z]``, produced by
        :meth:`VAE._encode` or :meth:`VAE.encode`: ``S`` samples from
        :math:`q(z \mid x)`, or the posterior mean (``S = 1``) with
        ``sample=False``, which is the default of :meth:`VAE.encode`.
    mu : torch.Tensor
        Mean of the approximate posterior :math:`q(z \mid x)`, shape ``[B, D_z]``.
    log_var : torch.Tensor
        Log-variance of :math:`q(z \mid x)`, shape ``[B, D_z]``.
    """

    z: torch.Tensor
    mu: torch.Tensor
    log_var: torch.Tensor

@dataclass(slots=True, repr=False)
class VAEDecodeOutput(ModelOutput):
    r"""Output of the VAE decoder stage.

    Attributes
    ----------
    x_hat : torch.Tensor
        Reconstructions or logits of shape ``[B, S, ...]``, produced by
        :meth:`VAE._decode` or :meth:`VAE.decode <pyautoencoder._base.BaseAutoencoder.decode>`.
    """

    x_hat: torch.Tensor

@dataclass(slots=True, repr=False)
class VAEOutput(ModelOutput):
    r"""Output of a full VAE forward pass.

    Attributes
    ----------
    x_hat : torch.Tensor
        Reconstructions or logits of shape ``[B, S, ...]``, produced by
        :meth:`VAE.forward`.
    z : torch.Tensor
        Latent codes of shape ``[B, S, D_z]``, produced by
        :meth:`VAE.forward`: ``S`` samples from :math:`q(z \mid x)`, or the
        posterior mean (``S = 1``) with ``sample=False``.
    mu : torch.Tensor
        Mean of :math:`q(z \mid x)`, shape ``[B, D_z]``.
    log_var : torch.Tensor
        Log-variance of :math:`q(z \mid x)`, shape ``[B, D_z]``.
    """

    x_hat: torch.Tensor
    z: torch.Tensor
    mu: torch.Tensor
    log_var: torch.Tensor

class VAE(BaseAutoencoder):
    r"""Variational Autoencoder following Kingma & Welling (2013).

    The model consists of:

    * an encoder mapping ``x → f(x)`` (feature representation),
    * a fully factorized Gaussian head producing ``(z, mu, log_var)``,
    * a decoder mapping latent samples ``z → x_hat``.

    Sampling is controlled explicitly by the ``sample`` argument, not by
    train/eval mode (which only affects layers such as dropout and batch
    normalization):

    * :meth:`forward` / :meth:`_encode` sample ``z`` from :math:`q(z \mid x)`
      by default (``sample=True``), as needed for training and for estimating
      the ELBO, also on validation data.
    * :meth:`encode` returns the posterior mean ``z = mu`` by default
      (``sample=False``), as usually needed at inference time.

    Example
    -------
    .. code-block:: python

        out = vae(x, S=5)                     # training: 5 samples per input
        loss = vae.compute_loss(x, out)

        vae.eval()
        with torch.no_grad():                 # validation ELBO
            val = vae.compute_loss(x_val, vae(x_val))

        enc = vae.encode(x)                   # inference: z = mu, [B, 1, D_z]
        x_rec = vae.decode(enc.z).x_hat       # deterministic reconstruction

    Shape convention
    ----------------
    Latent codes and reconstructions always carry an explicit Monte Carlo
    sample dimension ``S`` right after the batch dimension, **also when**
    ``S = 1``:

    * ``z``: ``[B, S, D_z]``
    * ``x_hat``: ``[B, S, ...]``
    * ``mu``, ``log_var``: ``[B, D_z]`` (one posterior per input, no ``S``)

    Consequently, :meth:`decode <pyautoencoder._base.BaseAutoencoder.decode>`
    expects ``z`` of shape ``[B, S, D_z]``. To decode a batch of single codes
    ``[N, D_z]`` (e.g. samples from the prior :math:`p(z)`), use
    ``z.unsqueeze(1)`` or call ``model.decoder`` directly.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        latent_dim: int,
    ):
        """Construct a Variational Autoencoder from an encoder, decoder, and latent size.

        Parameters
        ----------
        encoder : nn.Module
            Maps input ``x`` to a feature vector ``f(x)`` with shape ``[B, F]``.
        decoder : nn.Module
            Maps latent samples ``z`` to reconstructions ``x_hat``.
        latent_dim : int
            Dimensionality ``D_z`` of the latent space.

        Notes
        -----
        A :class:`~pyautoencoder.variational.stochastic_layers.FullyFactorizedGaussian`
        sampling layer is created internally as ``self.sampling_layer``; it is
        not a constructor parameter.
        """

        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.sampling_layer = stochastic_layers.FullyFactorizedGaussian(latent_dim=latent_dim)

    # --- training-time hooks required by BaseAutoencoder ---
    def _encode(self, x: torch.Tensor, S: int = 1, sample: bool = True) -> VAEEncodeOutput:
        r"""Encode inputs into posterior parameters and latent codes ``z``.

        Parameters
        ----------
        x : torch.Tensor
            Input batch of shape ``[B, ...]``. The encoder must output a flat
            feature vector per sample, of shape ``[B, F]``.
        S : int, optional
            Number of latent samples per input. Must be ``>= 1``, and ``1``
            when ``sample=False``. Defaults to 1.
        sample : bool, optional
            Whether to sample ``z`` from :math:`q(z \mid x)` (``True``) or to
            use the posterior mean ``z = mu`` (``False``). Defaults to ``True``.

        Returns
        -------
        VAEEncodeOutput
            Contains ``z`` of shape ``[B, S, D_z]`` (also for ``S = 1``), and
            ``mu`` and ``log_var`` of shape ``[B, D_z]``.

        Raises
        ------
        ValueError
            If ``S < 1``, if ``S != 1`` with ``sample=False``, or if the
            encoder output is not of shape ``[B, F]``.
        """

        f = self.encoder(x)
        z, mu, log_var = self.sampling_layer(f, S=S, sample=sample)
        return VAEEncodeOutput(z=z, mu=mu, log_var=log_var)

    @torch.no_grad()
    def encode(self, x: torch.Tensor, S: int = 1, sample: bool = False) -> VAEEncodeOutput:
        r"""Encode inputs without tracking gradients (inference API).

        Same as :meth:`_encode` under :func:`torch.no_grad`, but with
        an inference-friendly default: ``z`` is the posterior mean
        (``sample=False``). Pass ``sample=True`` to draw ``S`` samples.

        It does not change the train/eval mode: call ``model.eval()`` first
        if the encoder has layers such as dropout or batch normalization,
        otherwise the posterior mean is not deterministic.

        Parameters
        ----------
        x : torch.Tensor
            Input batch of shape ``[B, ...]``.
        S : int, optional
            Number of latent samples per input; must be ``1`` unless
            ``sample=True``. Defaults to 1.
        sample : bool, optional
            Whether to sample ``z`` from :math:`q(z \mid x)`. Defaults to
            ``False`` (``z = mu``).

        Returns
        -------
        VAEEncodeOutput
            Contains ``z`` of shape ``[B, S, D_z]`` (``[B, 1, D_z]`` by
            default), and ``mu`` and ``log_var`` of shape ``[B, D_z]``.

        Raises
        ------
        ValueError
            If ``S < 1``, if ``S != 1`` with ``sample=False``, or if the
            encoder output is not of shape ``[B, F]``.
        """

        return self._encode(x, S=S, sample=sample)

    def _decode(self, z: torch.Tensor) -> VAEDecodeOutput:
        """Decode latent variables into reconstructions.

        Parameters
        ----------
        z : torch.Tensor
            Latent samples of shape ``[B, S, D_z]``. The sample dimension
            ``S`` is required, also when ``S = 1``.

        Returns
        -------
        VAEDecodeOutput
            Contains ``x_hat`` of shape ``[B, S, ...]``.

        Raises
        ------
        ValueError
            If ``z`` is not of shape ``[B, S, D_z]`` with
            ``D_z == latent_dim``.
        """

        if z.dim() != 3 or z.size(-1) != self.latent_dim:
            raise ValueError(
                f"VAE expects z of shape [B, S, D_z] with D_z = {self.latent_dim} "
                f"(S = number of Monte Carlo samples, also when S = 1), got "
                f"{tuple(z.shape)}. For a batch of single codes [N, D_z] use "
                "z.unsqueeze(1), or call model.decoder directly."
            )

        B, S, D_z = z.shape
        x_hat_flat = self.decoder(z.reshape(B * S, D_z))  # [B * S, ...]
        x_hat = x_hat_flat.reshape(B, S, *x_hat_flat.shape[1:])
        return VAEDecodeOutput(x_hat=x_hat)

    def forward(self, x: torch.Tensor, S: int = 1, sample: bool = True) -> VAEOutput:
        r"""Full VAE pass: encode, draw ``S`` latent codes (or take the posterior mean), decode.

        Parameters
        ----------
        x : torch.Tensor
            Input batch of shape ``[B, ...]``.
        S : int, optional
            Number of latent samples for Monte Carlo estimates. Must be
            ``>= 1``, and ``1`` when ``sample=False``. Defaults to 1.
        sample : bool, optional
            Whether to sample ``z`` from :math:`q(z \mid x)` (``True``) or to
            use the posterior mean ``z = mu`` (``False``). Defaults to ``True``,
            in both train and eval mode.

        Returns
        -------
        VAEOutput
            Contains reconstructions ``x_hat`` of shape ``[B, S, ...]``,
            latent codes ``z`` of shape ``[B, S, D_z]``, and the posterior
            parameters ``mu`` and ``log_var`` of shape ``[B, D_z]``.

        Raises
        ------
        ValueError
            If ``S < 1``, if ``S != 1`` with ``sample=False``, or if the
            encoder output is not of shape ``[B, F]``.

        Notes
        -----
        If ``S > 1``, loss computation can broadcast ``x`` to shape
        ``[B, S, ...]`` without materializing copies. For Bernoulli likelihoods,
        the decoder must output logits.
        """

        enc = self._encode(x, S=S, sample=sample)
        dec = self._decode(enc.z)
        return VAEOutput(x_hat=dec.x_hat, z=enc.z, mu=enc.mu, log_var=enc.log_var)

    def compute_loss(self,
                     x: torch.Tensor,
                     output: VAEOutput,
                     beta: float = 1,
                     likelihood: str | LikelihoodType = LikelihoodType.GAUSSIAN) -> LossResult:
        r"""Compute the Evidence Lower Bound (ELBO) for a (beta-)Variational Autoencoder.

        This method implements the beta-VAE objective:

        .. math::

            \mathcal{L}(x; \beta)
                = \mathbb{E}_{q(z \mid x)}[\log p(x \mid z)]
                \;-\;
                \beta \, \mathrm{KL}(q(z \mid x) \,\|\, p(z)).

        The reconstruction term :math:`\log p(x \mid z)` is computed using
        :func:`~pyautoencoder.loss.log_likelihood`, which supports both
        Gaussian and Bernoulli likelihoods.

        Monte Carlo estimation
        ----------------------
        ``x_hat`` in ``output`` contains ``S`` Monte Carlo samples per input
        (``S = 1`` included), and the expectation
        :math:`\mathbb{E}_{q(z \mid x)}` is approximated by:

        .. math::

            \mathbb{E}_{q(z \mid x)}[\log p(x \mid z)]
                \approx \frac{1}{S} \sum_{s=1}^{S}
                \log p(x \mid z^{(s)}).

        Broadcasting
        ------------
        - ``x_hat`` must have shape ``[B, S, *x.shape[1:]]``: the dimension
          after the batch is always the sample dimension, also when ``S = 1``,
          and it is never inferred from the shapes.
        - ``x`` is broadcast over the sample dimension only, without copying
          data, so sample ``s`` of input ``i`` is always compared with ``x[i]``.

        Parameters
        ----------
        x : torch.Tensor
            Ground-truth inputs of shape ``[B, ...]``.
        output : VAEOutput
            Output from the VAE forward pass. Expected fields include:

            - ``x_hat`` (torch.Tensor): Reconstructed samples, shape ``[B, S, ...]``,
              where ``...`` is the shape of one input, ``x.shape[1:]``.
            - ``mu`` (torch.Tensor): Mean of :math:`q(z \mid x)`, shape ``[B, D_z]``
              with ``D_z = latent_dim``.
            - ``log_var`` (torch.Tensor): Log-variance of :math:`q(z \mid x)`, shape ``[B, D_z]``.

        beta : float, optional
            Weighting factor for the KL term (beta-VAE).
            ``beta = 1`` yields the standard VAE. Defaults to 1.
        likelihood : str | LikelihoodType, optional
            Likelihood model for the reconstruction term
            (``'gaussian'`` or ``'bernoulli'``). Defaults to Gaussian.

        Returns
        -------
        LossResult
            Result containing:

            * **objective** – Negative mean ELBO (scalar).
            * **diagnostics** – Dictionary of detached scalar tensors with:

              - ``"elbo"``: Batch mean of :math:`\mathcal{L}(x; \beta)`
                (the ELBO when ``beta = 1``).
              - ``"log_likelihood"``: Mean reconstruction term :math:`\mathbb{E}_{q}[\log p(x \mid z)]`.
              - ``"kl_divergence"``: Mean :math:`\mathrm{KL}(q \,\|\, p)` over the batch
                (not multiplied by ``beta``).

        Raises
        ------
        ValueError
            If ``x_hat`` is not of shape ``[B, S, *x.shape[1:]]`` with
            ``S >= 1``, if ``mu`` and ``log_var`` are not both of shape
            ``[B, latent_dim]``, or if ``likelihood`` is not a supported
            likelihood model.

        Notes
        -----
        - All returned diagnostics are **batch means**.
        - ``objective`` is not detached: gradients flow through ``x_hat``,
          ``mu`` and ``log_var`` to both the decoder and the encoder.
        - ``"elbo"`` is a Monte Carlo estimate of the ELBO only if ``output``
          was computed with ``sample=True`` (the default of :meth:`forward`,
          also in eval mode). With ``sample=False`` the reconstruction term is
          evaluated at ``z = mu``, which is not an ELBO estimate.
        - With the Gaussian likelihood, ``"elbo"`` and ``"log_likelihood"``
          omit the normalization constant of :math:`p(x \mid z)` (see
          :func:`~pyautoencoder.loss.log_likelihood`).
        """
        x_hat = output.x_hat
        mu = output.mu
        log_var = output.log_var

        if isinstance(likelihood, str):
            likelihood = LikelihoodType(likelihood.lower())

        # By contract dim 1 of x_hat is the sample dimension S, also when S = 1.
        # It is never inferred from the shapes: any guess is ambiguous and can
        # silently pair samples with the wrong elements of x.
        B = x.size(0)
        if (x_hat.dim() != x.dim() + 1
                or x_hat.shape[:1] + x_hat.shape[2:] != x.shape
                or x_hat.size(1) < 1):
            expected = ", ".join([str(B), "S", *map(str, x.shape[1:])])
            hint = (" If x_hat has no sample dimension, use x_hat.unsqueeze(1)."
                    if x_hat.dim() == x.dim() else "")
            raise ValueError(
                f"x_hat must have shape [B, S, *x.shape[1:]] = [{expected}] with "
                f"S >= 1, got {tuple(x_hat.shape)}. The decoder output for one "
                f"latent code must have the shape of one input.{hint}"
            )
        if mu.shape != (B, self.latent_dim) or log_var.shape != mu.shape:
            raise ValueError(
                f"mu and log_var must have shape [B, D_z] = [{B}, {self.latent_dim}], "
                f"got {tuple(mu.shape)} and {tuple(log_var.shape)}."
            )
        S = x_hat.size(1)

        # Broadcast x over the sample dimension only: [B, 1, ...] -> [B, S, ...].
        # The shapes above match everywhere else, and expand_as is a view (no copy).
        x_expanded = x.unsqueeze(1).expand_as(x_hat)

        # log p(x|z): elementwise -> sum over features => [B, S]
        log_px_z = log_likelihood(x_expanded, x_hat, likelihood=likelihood)
        log_px_z = log_px_z.reshape(B, S, -1).sum(-1)

        # E_q[log p(x|z)] via Monte Carlo average across S: [B]
        E_log_px_z = log_px_z.mean(dim=1)

        # KL(q||p): [B]
        kl_q_p = kl_divergence_diag_gaussian(mu, log_var)

        # ELBO per sample and batch means (retain grads)
        elbo_per_sample = E_log_px_z - beta * kl_q_p          # [B]
        elbo = elbo_per_sample.mean()                         # scalar

        return LossResult(
            objective = -elbo,
            diagnostics = {
                'elbo': elbo.detach(),
                'log_likelihood': E_log_px_z.mean().detach(),
                'kl_divergence': kl_q_p.mean().detach(),
            }
        )

@dataclass(slots=True, repr=False)
class AdaGVAEOutput(ModelOutput):
    r"""Output of a full AdaGVAE paired forward pass.

    Attributes
    ----------
    output1 : VAEOutput
        VAE output for the first input of the pair: built from the adapted
        posterior in training mode, plain :meth:`VAE.forward` output in eval
        mode.
    output2 : VAEOutput
        VAE output for the second input of the pair, as for ``output1``.
    """

    output1: VAEOutput
    output2: VAEOutput

class AdaGVAE(nn.Module):
    """Adaptive Group Variational Autoencoder (Ada-GVAE), from Locatello et al. (2020).

    Wraps a :class:`VAE` and adds adaptive posterior grouping for feature disentanglement.
    All VAE parameters are tracked through this wrapper.

    :meth:`forward` expects a pair of inputs ``(x1, x2)`` and returns an
    :class:`AdaGVAEOutput` for both. The adaptive grouping only changes
    training: in eval mode no grouping is applied and AdaGVAE behaves as the
    wrapped VAE, passing each input independently through :meth:`VAE.forward`.
    For single-image inference after training, use ``model.vae.encode`` and
    ``model.vae.decode`` directly.
    """

    def __init__(self, vae: VAE):
        """Wrap a VAE with adaptive posterior grouping.

        Parameters
        ----------
        vae : VAE
            A configured :class:`VAE` instance whose encoder, decoder, and sampling
            layer are reused for the paired training objective.
        """
        super().__init__()
        self.vae = vae

    def build(self, input_sample: torch.Tensor) -> None:
        """Materialize lazy layers by delegating to the wrapped VAE's build step.

        Parameters
        ----------
        input_sample : torch.Tensor
            A representative batch of single inputs (not a pair), passed to
            :meth:`VAE.build <pyautoencoder._base.BaseAutoencoder.build>`. Only its shape matters.
        """
        self.vae.build(input_sample)

    def _encode_pair(self, x1: torch.Tensor, x2: torch.Tensor, S: int = 1) -> tuple[VAEEncodeOutput, VAEEncodeOutput]:
        r"""Encode a pair of inputs with adaptive posterior alignment.

        As described in Locatello et al., this method:

        1. Encodes both inputs independently to obtain posterior parameters ``(mu1, log_var1)`` and ``(mu2, log_var2)``.
        2. Computes element-wise KL divergence between the two posteriors: ``KL(q1||q2) → [B, D_z]``.
        3. Computes a per-sample threshold ``tau = (max_d KL_d + min_d KL_d) / 2``.
        4. For each dimension, selects aligned (shared) or independent posteriors:

           - If ``KL(q1_d||q2_d) < tau``: uses the average posterior, whose
             mean and variance are the averages of the two means and of the
             two variances.
           - If ``KL(q1_d||q2_d) ≥ tau``: keeps the original posterior.

        5. Samples from the resulting (mixed) posteriors.

        Grouping and sampling are always applied, regardless of train/eval
        mode; :meth:`forward` only calls this method in train mode.

        Parameters
        ----------
        x1 : torch.Tensor
            First input batch of shape ``[B, ...]``.
        x2 : torch.Tensor
            Second input batch of shape ``[B, ...]``.
        S : int, optional
            Number of latent samples per input. Must be ``>= 1``. Defaults to 1.

        Returns
        -------
        tuple[VAEEncodeOutput, VAEEncodeOutput]
            A pair of ``VAEEncodeOutput`` objects, each containing:

            - ``z`` of shape ``[B, S, D_z]``: samples from the adapted posteriors.
            - ``mu`` of shape ``[B, D_z]``: the (adapted) means.
            - ``log_var`` of shape ``[B, D_z]``: the (adapted) log-variances.

        Notes
        -----
        The thresholding mechanism promotes learning of shared latent factors
        while allowing independent variation for high-divergence dimensions.
        This encourages disentanglement and structured representations.
        """
        mu1, log_var1 = self.vae.sampling_layer.get_params(self.vae.encoder(x1))
        mu2, log_var2 = self.vae.sampling_layer.get_params(self.vae.encoder(x2))

        # KL(q1||q2) -> [B, latents]
        kl_q1_q2 = kl_divergence_diag_gaussian(mu1, log_var1, mu2, log_var2, reduce_sum=False)

        # Computing threshold tau
        max_delta = torch.max(kl_q1_q2, dim=1, keepdim=True)[0]
        min_delta = torch.min(kl_q1_q2, dim=1, keepdim=True)[0]
        tau = 0.5 * (max_delta + min_delta)

        # Computing q_tilde1 and q_tilde2
        mu_mean = 0.5*(mu1 + mu2)
        var_mean = 0.5*(torch.exp(log_var1) + torch.exp(log_var2))
        log_var_mean = torch.log(var_mean)

        mask = kl_q1_q2 < tau
        mu_tilde1 = torch.where(mask, mu_mean, mu1)
        mu_tilde2 = torch.where(mask, mu_mean, mu2)
        log_var_tilde1 = torch.where(mask, log_var_mean, log_var1)
        log_var_tilde2 = torch.where(mask, log_var_mean, log_var2)

        z1 = self.vae.sampling_layer.reparametrize(mu=mu_tilde1, log_var=log_var_tilde1, S=S)
        z2 = self.vae.sampling_layer.reparametrize(mu=mu_tilde2, log_var=log_var_tilde2, S=S)

        return (
            VAEEncodeOutput(z=z1, mu=mu_tilde1, log_var=log_var_tilde1),
            VAEEncodeOutput(z=z2, mu=mu_tilde2, log_var=log_var_tilde2),
        )

    def forward(self, x: tuple[torch.Tensor, torch.Tensor], S: int = 1) -> AdaGVAEOutput:
        r"""AdaGVAE forward pass on a pair of inputs.

        * ``train()`` – adaptive grouping of the two posteriors, then sampling
          and decoding. Dimensions whose KL divergence
          :math:`\mathrm{KL}(q_1 \,\|\, q_2)` is below the per-sample threshold
          :math:`\tau = \tfrac{1}{2}(\max_d \mathrm{KL}_d + \min_d \mathrm{KL}_d)`
          share the average posterior (mean of the two means and of the two
          variances); the other dimensions keep their own posterior.
        * ``eval()`` – no grouping: each input goes independently through
          :meth:`VAE.forward`, which samples ``z`` from its own posterior.

        For single-image inference after training use ``model.vae.encode``
        and ``model.vae.decode``.

        Parameters
        ----------
        x : tuple[torch.Tensor, torch.Tensor]
            A ``(x1, x2)`` pair, each of shape ``[B, ...]``.
        S : int, optional
            Number of latent samples for Monte Carlo estimates. Must be
            ``>= 1``. Defaults to 1.

        Returns
        -------
        AdaGVAEOutput
            Pair outputs containing reconstructions and posterior parameters
            for both inputs (adapted posteriors in train mode).
        """
        x1, x2 = x
        if not self.training:
            return AdaGVAEOutput(output1=self.vae(x1, S=S), output2=self.vae(x2, S=S))

        x1_enc, x2_enc = self._encode_pair(x1, x2, S=S)
        x1_dec = self.vae._decode(x1_enc.z)
        x2_dec = self.vae._decode(x2_enc.z)
        return AdaGVAEOutput(
            output1=VAEOutput(x_hat=x1_dec.x_hat, z=x1_enc.z, mu=x1_enc.mu, log_var=x1_enc.log_var),
            output2=VAEOutput(x_hat=x2_dec.x_hat, z=x2_enc.z, mu=x2_enc.mu, log_var=x2_enc.log_var),
        )

    def compute_loss(self,
                     x: tuple[torch.Tensor, torch.Tensor],
                     output: AdaGVAEOutput,
                     beta: float = 1,
                     likelihood: str | LikelihoodType = LikelihoodType.GAUSSIAN) -> LossResult:
        r"""Compute the combined ELBO for a pair of inputs.

        In training mode the two terms use the adapted posteriors
        :math:`q(\hat{z} \mid x_i)` produced by :meth:`forward`; in eval mode
        they use the plain VAE posteriors.

        .. math::

            \mathcal{L}(x_1, x_2; \beta)
                = \left[ \mathbb{E}_{q(\hat{z} \mid x_1)}[\log p(x_1 \mid \hat{z})]
                \;-\; \beta \, \mathrm{KL}(q(\hat{z} \mid x_1) \,\|\, p(\hat{z})) \right]
                + \left[ \mathbb{E}_{q(\hat{z} \mid x_2)}[\log p(x_2 \mid \hat{z})]
                \;-\; \beta \, \mathrm{KL}(q(\hat{z} \mid x_2) \,\|\, p(\hat{z})) \right].

        Parameters
        ----------
        x : tuple[torch.Tensor, torch.Tensor]
            The ``(x1, x2)`` pair of ground-truth inputs, each of shape ``[B, ...]``.
        output : AdaGVAEOutput
            Output from :meth:`forward`.
        beta : float, optional
            KL weighting factor. ``beta = 1`` yields the standard objective.
            Defaults to 1.
        likelihood : str | LikelihoodType, optional
            Likelihood model for the reconstruction term (``'gaussian'`` or
            ``'bernoulli'``). Defaults to Gaussian.

        Returns
        -------
        LossResult
            Result containing:

            * **objective** – Sum of negative ELBOs for both inputs (scalar).
            * **diagnostics** – Dictionary of detached scalar tensors with:

              - ``"elbo"``: Sum of the two ``"elbo"`` diagnostics of
                :meth:`VAE.compute_loss` (weighted by ``beta``).
              - ``"log_likelihood_x1"``: Mean reconstruction term for ``x1``.
              - ``"log_likelihood_x2"``: Mean reconstruction term for ``x2``.
              - ``"kl_divergence_x1"``: Mean KL divergence for ``x1``.
              - ``"kl_divergence_x2"``: Mean KL divergence for ``x2``.

        Raises
        ------
        ValueError
            If either output of the pair fails the shape checks of
            :meth:`VAE.compute_loss`, or if ``likelihood`` is not a supported
            likelihood model.
        """
        x1, x2 = x
        loss1 = self.vae.compute_loss(x=x1, output=output.output1, beta=beta, likelihood=likelihood)
        loss2 = self.vae.compute_loss(x=x2, output=output.output2, beta=beta, likelihood=likelihood)
        return LossResult(
            objective=loss1.objective + loss2.objective,
            diagnostics={
                'elbo': loss1.diagnostics['elbo'] + loss2.diagnostics['elbo'],
                'log_likelihood_x1': loss1.diagnostics['log_likelihood'],
                'log_likelihood_x2': loss2.diagnostics['log_likelihood'],
                'kl_divergence_x1': loss1.diagnostics['kl_divergence'],
                'kl_divergence_x2': loss2.diagnostics['kl_divergence'],
            }
        )
