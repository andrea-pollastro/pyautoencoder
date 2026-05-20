Architecture & Design
=====================

Overview
--------

**PyAutoencoder** provides clean, well-documented implementations of the following
fundamental autoencoder architectures:

1. **Autoencoder (AE)** – Vanilla encoder-decoder pair
2. **Variational Autoencoder (VAE)** – Probabilistic encoder with sampling
3. **Adaptive Group VAE (AdaGVAE)** – Paired-input VAE for disentangled representations

All follow a consistent build/forward/compute_loss pattern and integrate naturally with
PyTorch. AdaGVAE extends the VAE interface for paired inputs.

Vanilla Autoencoder (AE)
------------------------

The :class:`~pyautoencoder.vanilla.AE` is a deterministic encoder-decoder model:

**Architecture**

- **Encoder** – Maps input :math:`x` to latent code :math:`z = f(x)`
- **Decoder** – Maps latent :math:`z` back to reconstruction :math:`\hat{x} = g(z)`
- **Forward pass** – Returns both :math:`z` and :math:`\hat{x}`

**Training**

.. code-block:: python

    from pyautoencoder.vanilla import AE

    model = AE(encoder=encoder_nn, decoder=decoder_nn)
    model.build(sample_input)

    # Training step
    output = model(x_batch)          # AEOutput with z and x_hat
    loss_result = model.compute_loss(x_batch, output, likelihood='bernoulli')
    loss_result.objective.backward()

**Inference**

.. code-block:: python

    # Extract latent codes
    z = model.encode(x_batch).z              # AEEncodeOutput → latent tensor

    # Reconstruct from latents
    x_reconstructed = model.decode(z).x_hat  # AEDecodeOutput → reconstruction tensor

**Output Structure**

.. code-block:: python

    class AEOutput:
        x_hat: torch.Tensor       # Reconstruction [B, ...]
        z: torch.Tensor           # Latent code [B, D_z]


Variational Autoencoder (VAE)
------------------------------

The :class:`~pyautoencoder.variational.VAE` implements the VAE framework
(Kingma & Welling, 2013) with probabilistic inference:

**Architecture**

- **Encoder** – Maps input :math:`x` to latent distribution :math:`q(z|x)`
- **Sampling layer** – Produces mean :math:`\mu`, log-variance :math:`\log\sigma^2`
- **Decoder** – Maps sampled :math:`z` to reconstruction distribution :math:`p(x|z)`

**Training**

.. code-block:: python

    from pyautoencoder.variational import VAE

    model = VAE(encoder=encoder_nn, decoder=decoder_nn, latent_dim=64)
    model.build(sample_input)

    # Training step (sample multiple times for Monte Carlo estimates)
    output = model(x_batch, S=5)  # 5 samples
    loss_result = model.compute_loss(x_batch, output, beta=1.0, likelihood='gaussian')
    loss_result.objective.backward()

**Training vs Evaluation**

- **Training mode** (:code:`model.train()`):
  - Samples :math:`S` latent codes per input
  - Enables Monte Carlo averaging of reconstruction loss
  - Returns shape :math:`[B, S, D_z]` for latents

- **Evaluation mode** (:code:`model.eval()`):
  - Deterministic output (uses means, no sampling)
  - Faster inference
  - Still computes :math:`\mu` and :math:`\log\sigma^2` for diagnostics

**Output Structure**

.. code-block:: python

    class VAEOutput:
        x_hat: torch.Tensor       # Reconstructions [B, S, ...]
        z: torch.Tensor           # Samples [B, S, D_z]
        mu: torch.Tensor          # Posterior mean [B, D_z]
        log_var: torch.Tensor     # Posterior log-variance [B, D_z]

**Sampling New Data**

Generate samples from the prior :math:`p(z) = \mathcal{N}(0, I)`:

.. code-block:: python

    with torch.no_grad():
        z_prior = torch.randn(n_samples, latent_dim)
        x_samples = model.decoder(z_prior)


Adaptive Group Variational Autoencoder (AdaGVAE)
-------------------------------------------------

The :class:`~pyautoencoder.variational.AdaGVAE` implements the Ada-GVAE framework
(Locatello et al., 2020). It wraps a :class:`~pyautoencoder.variational.VAE` and
adds an adaptive posterior-alignment step during training to encourage disentanglement.

**Architecture**

- **Backbone** – A fully configured :class:`~pyautoencoder.variational.VAE`
  (encoder, sampling layer, decoder are reused)
- **Paired encoding** – Both inputs :math:`x_1` and :math:`x_2` are encoded
  independently to obtain :math:`q_1(z|x_1)` and :math:`q_2(z|x_2)`
- **Adaptive alignment** – Dimensions where the per-dimension KL divergence
  :math:`\mathrm{KL}(q_1 \| q_2)` falls below a threshold :math:`\tau` are
  *shared* (posterior averaged); the rest are kept independent
- **Decoder** – The standard VAE decoder is applied to samples from the adapted
  posteriors

The threshold is computed per sample as the midpoint of the min and max
per-dimension KL values:

.. math::

    \tau = \tfrac{1}{2}(\max_d \mathrm{KL}_d + \min_d \mathrm{KL}_d)

**Training**

.. code-block:: python

    from pyautoencoder.variational import VAE, AdaGVAE

    vae = VAE(encoder=encoder_nn, decoder=decoder_nn, latent_dim=64)
    model = AdaGVAE(vae=vae)
    model.build(sample_input)

    # Training step — forward takes a pair
    output = model((x1_batch, x2_batch), S=5)
    loss_result = model.compute_loss((x1_batch, x2_batch), output, beta=4.0, likelihood='bernoulli')
    loss_result.objective.backward()

**Inference**

After training, AdaGVAE reuses the underlying VAE for single-image inference:

.. code-block:: python

    # Encode / decode through the wrapped VAE as usual
    z = model.vae.encode(x).z
    x_reconstructed = model.vae.decode(z).x_hat

**Output Structure**

.. code-block:: python

    class AdaGVAEOutput:
        output1: VAEOutput    # Adapted output for x1
        output2: VAEOutput    # Adapted output for x2

Each :class:`~pyautoencoder.variational.VAEOutput` contains ``x_hat``, ``z``,
``mu``, and ``log_var`` for the corresponding adapted posterior.


Loss Functions
--------------

The loss computation is integrated into each model via the :meth:`compute_loss` method.
This approach keeps loss logic close to the model implementation and ensures consistency.

Reconstruction Loss (AE)
~~~~~~~~~~~~~~~~~~~~~~~~

For standard autoencoders, use :meth:`AE.compute_loss` to compute reconstruction loss:

.. code-block:: python

    output = model(x)
    loss_result = model.compute_loss(x, output, likelihood='gaussian')
    loss_result.objective.backward()

Supported likelihoods:

- **Gaussian** – Continuous data
  
  .. math::
  
      \text{NLL} = \frac{1}{2}(x-\hat{x})^2

- **Bernoulli** – Discrete/binary data (logits)

  .. math::
  
      \text{NLL} = \text{BCE}_{\text{logits}}(x, \hat{x})


ELBO Loss (VAE)
~~~~~~~~~~~~~~~

For variational autoencoders, use :meth:`VAE.compute_loss` to compute the negative ELBO:

.. code-block:: python

    output = model(x, S=5)
    loss_result = model.compute_loss(x, output, beta=1.0, likelihood='gaussian')
    loss_result.objective.backward()

The ELBO decomposes into reconstruction and regularization terms:

.. math::

    \mathcal{L}(x; \beta) = \underbrace{\mathbb{E}_{q(z|x)}[\log p(x|z)]}_{\text{Reconstruction}}
        - \beta \, \underbrace{\mathrm{KL}(q(z|x) \| p(z))}_{\text{Regularization}}

**Beta-VAE Weighting**

The :math:`\beta` hyperparameter controls the KL weight:

- :math:`\beta = 1.0` – Standard VAE (matches true ELBO)
- :math:`\beta > 1.0` – Stronger regularization (more disentangled latents)
- :math:`\beta < 1.0` – Weaker regularization (better reconstruction)

Paired ELBO Loss (AdaGVAE)
~~~~~~~~~~~~~~~~~~~~~~~~~~

For AdaGVAE, use :meth:`AdaGVAE.compute_loss` to compute the combined objective
over the pair:

.. code-block:: python

    output = model((x1, x2), S=5)
    loss_result = model.compute_loss((x1, x2), output, beta=4.0, likelihood='bernoulli')
    loss_result.objective.backward()

The loss is the sum of two independent beta-VAE ELBOs, each evaluated on the
adapted posteriors :math:`q(\hat{z}|x_1)` and :math:`q(\hat{z}|x_2)`:

.. math::

    \mathcal{L}(x_1, x_2; \beta)
        = \Bigl[\mathbb{E}_{q(\hat{z}|x_1)}[\log p(x_1|\hat{z})]
          - \beta\,\mathrm{KL}(q(\hat{z}|x_1)\|p(\hat{z}))\Bigr]
        + \Bigl[\mathbb{E}_{q(\hat{z}|x_2)}[\log p(x_2|\hat{z})]
          - \beta\,\mathrm{KL}(q(\hat{z}|x_2)\|p(\hat{z}))\Bigr]

The diagnostics dictionary exposes per-input reconstruction and KL terms
(``log_likelihood_x1``, ``log_likelihood_x2``, ``kl_divergence_x1``,
``kl_divergence_x2``) so both views can be monitored independently.

**Loss Result Structure**

:meth:`AE.compute_loss`, :meth:`VAE.compute_loss`, and :meth:`AdaGVAE.compute_loss`
all return a :class:`~pyautoencoder.loss.LossResult` with:

- **objective** – scalar loss to optimize (backward-differentiable)
- **diagnostics** – dictionary of scalar metrics (float values)

Example:

.. code-block:: python

    loss_result = model.compute_loss(x, output)
    loss_result.objective.backward()              # Optimize this
    for name, val in loss_result.diagnostics.items():
        log(name, val)                            # Monitor these


Key Design Principles
----------------------

**User-Provided Networks**

You provide the encoder and decoder as arbitrary PyTorch modules:

.. code-block:: python

    encoder = my_custom_encoder_network()  # Any nn.Module
    decoder = my_custom_decoder_network()  # Any nn.Module
    model = AE(encoder, decoder)

This keeps the library flexible and composable with existing PyTorch code.

**Explicit Initialization**

The :meth:`build` method initializes size-dependent parameters:

.. code-block:: python

    model.build(sample_input)  # Required once before training

This catches shape mismatches early and makes model behavior transparent.

**Clear Loss Integration**

Loss functions are methods on the models themselves, providing a clean API:

.. code-block:: python

    output = model(x_batch)
    loss_result = model.compute_loss(x_batch, output)
    loss_result.objective.backward()              # Optimize this
    for name, val in loss_result.diagnostics.items():
        log(name, val)                            # Monitor these

**Consistent Interfaces**

All the architectures follow the same pattern:

.. code-block:: python

    model.build(sample)                          # Initialize
    output = model(x_batch)                      # Forward pass
    loss_result = model.compute_loss(x, output)  # Compute loss
    loss_result.objective.backward()             # Backprop


