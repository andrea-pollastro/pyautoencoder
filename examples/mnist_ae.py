# --- Standard Autoencoder on MNIST ---

import random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from pyautoencoder.vanilla import AE
from pyautoencoder.loss import AELoss

# ---------------- Config ----------------
LATENTS = 128
EPOCHS = 30
BATCH_SIZE = 128
LR = 0.001
NROWS_SHOW = 3

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 1926
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---------------- Data ------------------
tfm = transforms.ToTensor()  # in [0,1]
train_ds = datasets.MNIST("./data", train=True,  download=True, transform=tfm)
test_ds  = datasets.MNIST("./data", train=False, download=True, transform=tfm)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

# ---------------- Model -----------------
def make_ae(latent_dim):
    encoder = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, latent_dim),
        nn.ReLU(),
    )
    decoder = nn.Sequential(
        nn.Linear(latent_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 512),
        nn.ReLU(),
        nn.Linear(512, 28*28),
        nn.Unflatten(-1, (1, 28, 28))  # keep last layer linear; VAELoss(bernoulli) handles logits
    )
    model = AE(encoder=encoder, decoder=decoder)
    model.build(input_sample=torch.randn(1, 1, 28, 28))
    return model

# Reconstruction loss
loss_fn = AELoss(likelihood='bernoulli')

# ---------------- Train -----------------
def train_epochs(model: nn.Module, epochs: int):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    model.train()
    t0 = time.time()
    for ep in range(1, epochs + 1):
        epoch_loss, n = 0.0, 0
        for x, _ in train_loader:
            x = x.to(DEVICE)
            opt.zero_grad()
            out = model(x)
            loss_info = loss_fn(x, out)
            loss_info.total.backward()
            opt.step()
            epoch_loss += loss_info.total.item() * x.size(0)
            n += x.size(0)
        print(f"Epoch {ep:3d}/{epochs}  NLL={epoch_loss/n:.4f}  (+{time.time()-t0:.1f}s)")

# ---------------- Plot (TEST samples) ---
@torch.no_grad()
def plot_test_recons(model: nn.Module, nz: int, nrows: int = NROWS_SHOW, fname: str | None = None):
    model.eval()
    x_batch, _ = next(iter(test_loader))
    idx = torch.randperm(x_batch.size(0))[:nrows]
    x = x_batch[idx].to(DEVICE)

    out = model(x)
    x_hat = torch.sigmoid(out.x_hat)

    x = x.cpu().numpy()
    x_hat = x_hat.cpu().numpy()

    fig, axes = plt.subplots(nrows, 2, figsize=(4.5, 2.5 * nrows))
    if nrows == 1:
        axes = np.array([axes])

    axes[0, 0].set_title("Original")
    axes[0, 1].set_title(f"Reconstruction (Nz={nz})")

    for r in range(nrows):
        axes[r, 0].imshow(x[r, 0], cmap='gray', vmin=0, vmax=1)
        axes[r, 0].axis('off')
        axes[r, 1].imshow(x_hat[r, 0], cmap='gray', vmin=0, vmax=1)
        axes[r, 1].axis('off')

    plt.tight_layout()
    out = fname or f"ae_mnist_test_recon_nz{nz}.png"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved figure -> {out}")

# ---------------- Run -------------------
if __name__ == "__main__":
    print(f"\n=== Training AE (latent_dim={LATENTS}) for {EPOCHS} epochs ===")
    model = make_ae(LATENTS)
    train_epochs(model, EPOCHS)
    plot_test_recons(model, LATENTS, nrows=NROWS_SHOW, fname=f"ae_mnist_test_recon_nz{LATENTS}.png")
    print("Done.")
