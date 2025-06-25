# pyautoencoders

**pyautoencoders** is a lightweight PyTorch package offering clean, minimal implementations of foundational autoencoder architectures. 
It is designed for researchers, educators, and practitioners seeking a reliable base for experimentation, extension, or instruction.

## 📦 Installation

```bash
pip install pyautoencoders
```

Or install from source:
```bash
git clone https://github.com/andrea-pollastro/pyautoencoders.git
cd pyautoencoders
pip install -e .
```

## 🚀 Quick Example

```python
import torch
from pyautoencoders.models import Autoencoder

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
@misc{pollastro:2025,
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
