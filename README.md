# PyAutoencoder

A clean, modular PyTorch library for building and training autoencoders.

<!-- <p align="center">
  <img src="assets/logo_nobackground.png" alt="pyautoencoder logo" width="220"/>
</p> -->

![logo](https://raw.githubusercontent.com/andrea-pollastro/pyautoencoder/main/assets/logo_nobackground.png)

<p align="center">
  <!-- PyPI Version -->
  <a href="https://pypi.org/project/pyautoencoder/">
    <img alt="PyPI Version" src="https://img.shields.io/pypi/v/pyautoencoder.svg">
  </a>

  <!-- Python Versions -->
  <img alt="Python Versions" src="https://img.shields.io/pypi/pyversions/pyautoencoder.svg">

  <!-- CI Status -->
  <a href="https://github.com/andrea-pollastro/pyautoencoder/actions/workflows/ci.yml">
    <img alt="CI" src="https://github.com/andrea-pollastro/pyautoencoder/actions/workflows/ci.yml/badge.svg">
  </a>

  <!-- Publish Status -->
  <a href="https://github.com/andrea-pollastro/pyautoencoder/actions/workflows/publish.yml">
    <img alt="Publish to PyPI" src="https://github.com/andrea-pollastro/pyautoencoder/actions/workflows/publish.yml/badge.svg">
  </a>

  <!-- License -->
  <a href="https://github.com/andrea-pollastro/pyautoencoder/blob/main/LICENSE">
    <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-blue.svg">
  </a>

  <!-- GitHub Stars -->
  <a href="https://github.com/andrea-pollastro/pyautoencoder/stargazers">
    <img alt="Stars" src="https://img.shields.io/github/stars/andrea-pollastro/pyautoencoder?style=social">
  </a>

  <!-- Downloads (optional but nice) -->
  <a href="https://pepy.tech/project/pyautoencoder">
    <img alt="Downloads" src="https://pepy.tech/badge/pyautoencoder">
  </a>
</p>


---

## Highlights

PyAutoencoder is designed to offer **simple and easy access to autoencoder frameworks**. Here's what it offers:

- **Minimal, composable API**  
You don't have to inherit from complicated base classes or learn a new training loop. Simply provide your own PyTorch nn.Module encoder and decoder, and plug them into the ready‑to‑use autoencoder wrappers. This makes it easy to experiment with different architectures (e.g. MLPs, CNNs) while reusing the same training pipeline.

- **Ready‑to‑use autoencoders**
  The library ships with working implementations of autoencoders, each paired with their respective loss functions. You can start training in a few lines, without re‑implementing reconstruction likelihoods, KL divergence, or other boilerplate.

- **PyTorch compatibility**  
  The library is fully compatible with the PyTorch ecosystem, so models integrate naturally with modules, tensors, optimizers, and schedulers.

- **Lightweight, research‑oriented**  
  The library is intentionally minimal: no training loop frameworks, no heavy abstractions. This makes it well suited for research prototypes where you want control and transparency.

> **Status**: The project is in an early but usable stage. Contributions, issues, and feedback are highly encouraged!

**Currently implemented**:
- Autoencoder (AE)
- Variational Autoencoder (VAE)

---

## Installation

```bash
pip install pyautoencoder
```

Or install from source for development:

```bash
git clone https://github.com/andrea-pollastro/pyautoencoder.git
cd pyautoencoder
pip install -e .
```

## Quick start

```python
import torch
import torch.nn as nn
from pyautoencoder.variational import VAE
from pyautoencoder.loss import VAELoss

# Define encoder/decoder
encoder = nn.Sequential(
    nn.Linear(784, 512), 
    nn.ReLU(),
    nn.Linear(512, 256)
)

decoder = nn.Sequential(
    nn.Linear(256, 512), 
    nn.ReLU(),
    nn.Linear(512, 784)
)

# Model
vae = VAE(encoder=encoder, decoder=decoder, latent_dim=32)

# Loss
criterion = VAELoss(beta=1.0, likelihood="gaussian")
optimizer = torch.optim.Adam(vae.parameters())
for x in dataloader:
    optimizer.zero_grad()
    out = vae(x)
    losses = criterion(out, x)
    losses.total.backward() # negative ELBO
    optimizer.step()

    # optional: log components
    log_likelihood = losses.components["log_likelihood"]
    kl_divergence = losses.components["kl_divergence"]
```

## Built‑in models

- **`AE`** — standard Autoencoder
  ```python
  from pyautoencoder.vanilla import AE
  from pyautoencoder.loss import AELoss
  ae = AE(encoder=encoder, decoder=decoder)
  criterion = AELoss(likelihood="gaussian") # or bernoulli
  ```

- **`VAE`** — Variational Autoencoder
  ```python
  from pyautoencoder.variational import VAE, VAELoss
  from pyautoencoder.loss import VAELoss
  vae = VAE(encoder=encoder, decoder=decoder, latent_dim=32)
  criterion = VAELoss(beta=1.0, likelihood="gaussian") # or bernoulli
  ```

## Examples

See the [`examples/`](examples/) folder for runnable scripts showing example of usage.

## License

This project is released under the **MIT License**. See [LICENSE](LICENSE).

## Citation

If you use this package in academic work, please cite:

```bibtex
@misc{pollastro2025pyautoencoder,
  author       = {Andrea Pollastro},
  title        = {pyautoencoder},
  year         = {2025},
  howpublished = {GitHub repository},
  url          = {https://github.com/andrea-pollastro/pyautoencoder}
}
```
