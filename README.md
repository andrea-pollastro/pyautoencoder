![logo](https://raw.githubusercontent.com/andrea-pollastro/pyautoencoder/main/assets/logo_nobackground.png)
[![PyPI version](https://img.shields.io/pypi/v/pyautoencoder.svg?color=orange&label=pypi)](https://pypi.org/project/pyautoencoder/)

# PyAutoencoder

A clean, modular PyTorch library for autoencoder models.

## Installation

```bash
pip install pyautoencoder
```

## Quick Start

```python
import torch
import torch.nn as nn
from pyautoencoder import VAE, VAELoss

# Define your encoder and decoder networks
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

# Create VAE model
vae = VAE(
    encoder=encoder,
    decoder=decoder,
    latent_dim=32
)

# Create loss function
vae_loss = VAELoss(beta=1.0)  # beta=1.0 for standard VAE

# Define encoder and decoder networks
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

# Create VAE
vae = VAE(
    encoder=encoder,
    decoder=decoder,
    latent_dim=32
)

# Create loss function
vae_loss = VAELoss(beta=1.0, likelihood='gaussian')

# Training loop example
optimizer = torch.optim.Adam(vae.parameters())

for batch in dataloader:
    optimizer.zero_grad()
    output = vae(batch)
    loss_info = vae_loss(output, batch)
    
    # Access total loss and components
    total_loss = loss_info.total
    recon_loss = loss_info.components['reconstruction']
    kl_loss = loss_info.components['kl_divergence']
    
    total_loss.backward()
    optimizer.step()
```

## Available Models

### Autoencoder (AE)
```python
from pyautoencoder import AE, AutoencoderLoss

ae = AE(encoder=encoder, decoder=decoder)
ae_loss = AutoencoderLoss(likelihood='gaussian')
```

### Variational Autoencoder (VAE)
```python
from pyautoencoder import VAE, VAELoss

vae = VAE(encoder=encoder, decoder=decoder, latent_dim=32)
vae_loss = VAELoss(beta=1.0, likelihood='gaussian')
```

## Loss Components

All losses return a `LossComponents` object containing:
- `total`: The total loss to use for backpropagation
- `components`: Dictionary of individual loss components for monitoring

Example:
```python
loss_info = vae_loss(output, x)
total_loss = loss_info.total
recon_loss = loss_info.components['reconstruction']
kl_loss = loss_info.components['kl_divergence']
```

## Contributing

Contributions are welcome! Please check out our contribution guidelines.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
[![License](https://img.shields.io/github/license/andrea-pollastro/pyautoencoder.svg)](https://opensource.org/licenses/MIT)

## 📦 Installation

```bash
pip install pyautoencoder
```

Or install from source:
```bash
git clone https://github.com/andrea-pollastro/pyautoencoder.git
cd pyautoencoder
pip install -e .
```

## 🤝 Contributing
Contributions are welcome — especially new autoencoder variants, training examples, and documentation improvements.
Please open an issue or pull request to discuss any changes.

## 📝 Citing
```bibtex
@misc{pollastro2025pyautoencoder,
  Author = {Andrea Pollastro},
  Title = {pyautoencoder},
  Year = {2025},
  Publisher = {GitHub},
  Journal = {GitHub repository},
  Howpublished = {\url{https://github.com/andrea-pollastro/pyautoencoder}}
}
```

## 📄 License
This project is licensed under the MIT License. See the LICENSE file for details.
