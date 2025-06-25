![logo](https://raw.githubusercontent.com/andrea-pollastro/pyautoencoder/main/assets/logo_nobackground.png)
[![PyPI version](https://img.shields.io/pypi/v/pyautoencoder.svg?color=orange&label=pypi)](https://pypi.org/project/pyautoencoder/)
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

## 🚀 Quick Example

```python
import torch
from pyautoencoder.models import Autoencoder

# Define encoder and decoder
encoder = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(784, 32)
)

decoder = torch.nn.Sequential(
    torch.nn.Linear(32, 784),
    torch.nn.Unflatten(1, (1, 28, 28))
)

# Initialize model
model = Autoencoder(encoder, decoder)

# Forward pass
x = torch.randn(64, 1, 28, 28)
x_hat, z = model(x)
```

## 🗺️ Roadmap
- [x] Autoencoder (AE)
- [x] Variational Autoencoder (VAE)
- [ ] Hierarchical VAE (HVAE)
- [ ] Importance-Weighted AE (IWAE)
- [ ] Denoising Autoencoder (DAE)
- [ ] Sparse Autoencoder (SAE)

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
