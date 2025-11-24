Getting Started
===============

**pyautoencoder** is a PyTorch library for building and training state-of-the-art autoencoders. 
It is designed to host a growing collection of autoencoder variants, with structured outputs and modular 
loss functions to make experimentation easy.

Installation
------------

Install **pyautoencoder** via pip:

.. code-block:: bash

    pip install pyautoencoder

Or install from source:

.. code-block:: bash

    git clone https://github.com/andrea-pollastro/pyautoencoders.git
    cd pyautoencoders
    pip install -e .

Quick Start
-----------

Vanilla Autoencoder (AE)
~~~~~~~~~~~~~~~~~~~~~~~~

Train a simple autoencoder on MNIST:

.. code-block:: python

    import torch
    import torch.nn as nn
    from pyautoencoder.vanilla import AE
    from pyautoencoder.loss import AELoss

    # Define encoder and decoder
    encoder = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 64),
    )

    decoder = nn.Sequential(
        nn.Linear(64, 256),
        nn.ReLU(),
        nn.Linear(256, 784),
    )

    # Create and build the model
    model = AE(encoder=encoder, decoder=decoder)
    model.build(torch.randn(1, 1, 28, 28))

    # Training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = AELoss(likelihood='gaussian')

    x_batch = torch.randn(32, 1, 28, 28)
    output = model(x_batch)
    loss_components = loss_fn(x_batch, output)
    optimizer.zero_grad()
    loss_components.total.backward()
    optimizer.step()

    # Inference on z and x_hat (no gradients)
    with torch.no_grad():
        z = model.encode(x_batch)
        x_hat = model.decode(z)

Variational Autoencoder (VAE)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Train a VAE with the reparameterization trick:

.. code-block:: python

    import torch
    import torch.nn as nn
    from pyautoencoder.variational import VAE
    from pyautoencoder.loss import VAELoss

    # Define encoder and decoder
    encoder = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
    )

    decoder = nn.Sequential(
        nn.Linear(64, 256),
        nn.ReLU(),
        nn.Linear(256, 784),
    )

    # Create and build the model
    model = VAE(encoder=encoder, decoder=decoder, latent_dim=64)
    model.build(torch.randn(1, 1, 28, 28))

    # Training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = VAELoss(beta=1.0, likelihood='gaussian')

    x_batch = torch.randn(32, 1, 28, 28)
    output = model(x_batch, S=1)  # S=1: single Monte Carlo sample
    loss_components = loss_fn(x_batch, output)
    optimizer.zero_grad()
    loss_components.total.backward()
    optimizer.step()

    # Print diagnostics
    print(f"ELBO: {loss_components.metrics['elbo']:.4f}")
    print(f"NLL/dim (bits): {loss_components.metrics['nll_per_dim_bits']:.4f}")
    print(f"KL/latent-dim (nats): {loss_components.metrics['beta_kl_per_latent_dim_nats']:.4f}")

Core Concepts
-------------

ModelOutput
~~~~~~~~~~~

All forward passes return structured output dataclasses that are easy to inspect:

.. code-block:: python

    output = model(x)
    print(output)
    # AEOutput(x_hat=Tensor(shape=(32, 1, 28, 28), dtype=torch.float32),
    #          z=Tensor(shape=(32, 64), dtype=torch.float32))

Loss Components
~~~~~~~~~~~~~~~

Loss functions return structured results with optimization targets and diagnostics:

.. code-block:: python

    loss_components = loss_fn(x, model_output)
    loss_components.total.backward()      # Optimize this
    print(loss_components.metrics)        # Log these for monitoring

