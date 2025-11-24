Architecture & Design
=====================

Overview
--------

**PyAutoencoder** provides clean, well-documented implementations of the following
fundamental autoencoder architectures:

1. **Standard Autoencoder (AE)** – Deterministic encoder-decoder pair
2. **Variational Autoencoder (VAE)** – Probabilistic encoder with sampling

Both follow consistent interfaces and integrate naturally with PyTorch.

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
    from pyautoencoder.loss import AELoss

    model = AE(encoder=encoder_nn, decoder=decoder_nn)
    model.build(sample_input)

    # Training step
    output = model(x_batch)          # [z, x_hat]
    loss_fn = AELoss(likelihood='gaussian')
    loss = loss_fn(x_batch, output)
    loss.total.backward()

**Inference**

.. code-block:: python

    # Extract latent codes
    z = model.encode(x_batch)

    # Reconstruct from latents
    x_reconstructed = model.decode(z)

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
    from pyautoencoder.loss import VAELoss

    model = VAE(encoder=encoder_nn, decoder=decoder_nn, latent_dim=64)
    model.build(sample_input)

    # Training step (sample multiple times for Monte Carlo estimates)
    output = model(x_batch, S=5)  # 5 samples
    loss_fn = VAELoss(beta=1.0, likelihood='gaussian')
    loss = loss_fn(x_batch, output)
    loss.total.backward()

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

Loss Functions
--------------

Reconstruction Loss (AE)
~~~~~~~~~~~~~~~~~~~~~~~~

For standard autoencoders, the loss is the reconstruction negative log-likelihood:

.. code-block:: python

    loss_fn = AELoss(likelihood='gaussian')
    loss = loss_fn(x, ae_output)

Supported likelihoods:

- **Gaussian** – Continuous data
  
  .. math::
  
      \text{NLL} = \frac{1}{2}[(x-\hat{x})^2 + \log(2\pi)]

- **Bernoulli** – Discrete/binary data (logits)

  .. math::
  
      \text{NLL} = \text{BCE}_{\text{logits}}(x, \hat{x})


ELBO Loss (VAE)
~~~~~~~~~~~~~~~

For variational autoencoders, the loss is the negative ELBO:

.. code-block:: python

    loss_fn = VAELoss(beta=1.0, likelihood='gaussian')
    loss = loss_fn(x, vae_output)

The ELBO decomposes into reconstruction and regularization terms:

.. math::

    \mathcal{L}(x; \beta) = \underbrace{\mathbb{E}_{q(z|x)}[\log p(x|z)]}_{\text{Reconstruction}}
        - \beta \, \underbrace{\mathrm{KL}(q(z|x) \| p(z))}_{\text{Regularization}}

**Beta-VAE Weighting**

The :math:`\beta` hyperparameter controls the KL weight:

- :math:`\beta = 1.0` – Standard VAE (matches true ELBO)
- :math:`\beta > 1.0` – Stronger regularization (more disentangled latents)
- :math:`\beta < 1.0` – Weaker regularization (better reconstruction)

**Loss Output**

Both :class:`~pyautoencoder.loss.AELoss` and :class:`~pyautoencoder.loss.VAELoss`
return a :class:`~pyautoencoder.loss.LossComponents` with:

- **total** – scalar loss to optimize
- **components** – named loss terms
- **metrics** – diagnostics (bits/dim, KL/latent, ELBO, etc.)

Example:

.. code-block:: python

    loss = loss_fn(x, output)
    print(loss.total)                                   # Tensor to optimize
    print(loss.components)                              # {'nll': ..., 'kl': ...}
    print(loss.metrics['nll_per_dim_bits'])             # Bits per input dimension
    print(loss.metrics['beta_kl_per_latent_dim_nats'])  # Nats per latent


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

**Clear Loss Breakdown**

Loss functions expose component terms and metrics for transparency:

.. code-block:: python

    loss = loss_fn(x, output)
    loss.total.backward()              # Optimize this
    for name, val in loss.metrics.items():
        log(name, val)                 # Monitor these

**Consistent Interfaces**

All the architectures follow the same pattern:

.. code-block:: python

    model.build(sample)              # Initialize
    output = model(x_batch)          # Forward pass
    loss = loss_fn(x_batch, output)  # Compute loss
    loss.total.backward()            # Backprop


