# PyAutoencoder

A clean, modular PyTorch-based library for building and training autoencoders.

![logo](https://raw.githubusercontent.com/andrea-pollastro/pyautoencoder/main/assets/logo_nobackground.png)

<p align="center">
  <a href="https://pypi.org/project/pyautoencoder/"><img alt="PyPI Version" src="https://img.shields.io/pypi/v/pyautoencoder.svg"></a>
  <!-- <img alt="Python Versions" src="https://img.shields.io/pypi/pyversions/pyautoencoder.svg"> -->
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-blue">
  <a href="https://github.com/andrea-pollastro/pyautoencoder/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/andrea-pollastro/pyautoencoder/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://github.com/andrea-pollastro/pyautoencoder/actions/workflows/publish.yml"><img alt="Publish to PyPI" src="https://github.com/andrea-pollastro/pyautoencoder/actions/workflows/publish.yml/badge.svg"></a>
  <a href="https://github.com/andrea-pollastro/pyautoencoder/blob/main/LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-blue.svg"></a>
  <a href="https://pepy.tech/project/pyautoencoder"><img alt="Downloads" src="https://pepy.tech/badge/pyautoencoder"></a>
  <a href="https://pyautoencoder.readthedocs.io/en/latest/"><img alt="Documentation Status" src="https://readthedocs.org/projects/pyautoencoder/badge/?version=latest"></a>
  <a href="https://github.com/andrea-pollastro/pyautoencoder/stargazers"><img alt="Stars" src="https://img.shields.io/github/stars/andrea-pollastro/pyautoencoder?style=social"></a>
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

## Documentation

Full documentation (installation, tutorials, API reference, and examples) is available at:

👉 **https://pyautoencoder.readthedocs.io/en/latest/**

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

# Define encoder/decoder
latent_dim = 32
encoder = nn.Sequential(
    nn.Linear(784, 512), 
    nn.ReLU(),
)

decoder = nn.Sequential(
    nn.Linear(latent_dim, 512), 
    nn.ReLU(),
    nn.Linear(512, 784)
)

# Model
vae = VAE(encoder=encoder, decoder=decoder, latent_dim=latent_dim)

optimizer = torch.optim.Adam(vae.parameters())
for x in dataloader:
    optimizer.zero_grad()
    out = vae(x)
    loss_results = vae.compute_loss(out, x, beta=1, likelihood='bernoulli')
    loss_results.objective.backward() # negative ELBO
    optimizer.step()
    # optional: log components
    log_likelihood = loss_results.components["log_likelihood"]
    kl_divergence = loss_results.components["kl_divergence"]
```

## Examples

The [`examples/`](examples/) directory contains runnable scripts, including:

- **`mnist_ae.py`** – standard Autoencoder on MNIST  
- **`mnist_vae_kingma2013.py`** – reproduction of the MNIST VAE experiment from **Kingma & Welling (2013), Fig. 2**

These examples are also documented and explained in the [online documentation](https://pyautoencoder.readthedocs.io/en/latest/).

---

## License

This project is released under the **MIT License**. See [LICENSE](LICENSE).

## Citation

If you use this package in academic work, please cite:

```bibtex
@article{pollastro2025sincvae,
  title={SincVAE: A new semi-supervised approach to improve anomaly detection on EEG data using SincNet and variational autoencoder},
  author={Pollastro, Andrea and Isgr{\`o}, Francesco and Prevete, Roberto},
  journal={Computer Methods and Programs in Biomedicine Update},
  pages={100213},
  year={2025},
  publisher={Elsevier}
}
```
