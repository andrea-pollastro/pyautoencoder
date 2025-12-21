from __future__ import annotations

from pathlib import Path
import time
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from pyautoencoder.vanilla import AE


# ---------------- Config ---------------- #
LATENT_DIM = 128
NUM_EPOCHS = 30
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
NUM_RECON_COLS = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 1926

# ---------------- Utils ----------------- #
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


# ---------------- Data ------------------ #
def make_dataloaders(batch_size: int) -> tuple[DataLoader, DataLoader]:
    tfm = transforms.ToTensor()  # maps to [0,1]
    train_dataset = datasets.MNIST("./data", train=True,  download=True, transform=tfm)
    test_dataset  = datasets.MNIST("./data", train=False, download=True, transform=tfm)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)
    return train_dataloader, test_dataloader


# ---------------- Model ----------------- #
def make_autoencoder(latent_dim: int) -> AE:
    encoder = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 256),
        nn.ReLU(),
        nn.Linear(256, latent_dim),
    )
    decoder = nn.Sequential(
        nn.Linear(latent_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 28 * 28),
        nn.Unflatten(-1, (1, 28, 28)),  # keep last layer linear; AELoss(bernoulli) expects logits
    )
    model = AE(encoder=encoder, decoder=decoder)
    model.build(input_sample=torch.randn(1, 1, 28, 28))
    return model

# ---------------- Train ----------------- #
def train(
    model: AE,
    train_loader: DataLoader,
    num_epochs: int,
    lr: float = LEARNING_RATE,
) -> None:
    model.to(DEVICE)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    start_time = time.time()
    for epoch in range(1, num_epochs + 1):
        running_log_likelihood = 0.0
        n = 0

        for x, _ in train_loader:
            x = x.to(DEVICE)
            optimizer.zero_grad()

            out = model(x)
            loss_info = model.compute_loss(x, out, likelihood='bernoulli')
            loss_info.objective.backward()
            optimizer.step()

            batch_size = x.size(0)
            running_log_likelihood += loss_info.diagnostics['log_likelihood'] * batch_size
            n += batch_size

        avg_NLL = running_log_likelihood / n
        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch:3d}/{num_epochs}  "
            f"LogLikelihood={avg_NLL:.4f} | "
            f"(elapsed {elapsed:.1f}s)"
        )


# ---------------- Plot (TEST samples) ---------------- #
@torch.no_grad()
def plot_test_reconstructions(
    model: AE,
    test_loader: DataLoader,
    latent_dim: int,
    num_cols: int = NUM_RECON_COLS,
    fname: str | None = None,
) -> None:
    model.eval()

    x_batch, _ = next(iter(test_loader))
    idx = torch.randperm(x_batch.size(0))[:num_cols]
    x = x_batch[idx].to(DEVICE)

    out = model(x)
    x_hat = torch.sigmoid(out.x_hat)

    x = x.cpu().numpy()
    x_hat = x_hat.cpu().numpy()

    fig, axes = plt.subplots(2, num_cols, figsize=(2.5 * num_cols, 4.5))
    if num_cols == 1:
        axes = np.array([axes])

    axes[0, 0].set_ylabel("Original", fontsize=18)
    axes[1, 0].set_ylabel("Reconstruction", fontsize=18)

    for r in range(num_cols):
        axes[0, r].imshow(x[r, 0], cmap="gray", vmin=0, vmax=1)
        axes[0, r].set_xticks([])
        axes[0, r].set_yticks([])
        axes[1, r].imshow(x_hat[r, 0], cmap="gray", vmin=0, vmax=1)
        axes[1, r].set_xticks([])
        axes[1, r].set_yticks([])

    plt.tight_layout()
    out_path = fname or f"ae_mnist_test_recon_latent{latent_dim}.png"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure -> {out_path}")


# ---------------- Main ------------------ #
def main() -> None:
    print(f"\n=== Training AE (latent_dim={LATENT_DIM}) for {NUM_EPOCHS} epochs ===")
    print(f"Using device: {DEVICE}")

    set_seed(SEED)
    train_loader, test_loader = make_dataloaders(BATCH_SIZE)

    model = make_autoencoder(LATENT_DIM)
    train(model, train_loader, NUM_EPOCHS)
    plot_test_reconstructions(
        model,
        test_loader,
        latent_dim=LATENT_DIM,
        num_cols=NUM_RECON_COLS,
        fname=f"ae_mnist_test_recon_latent{LATENT_DIM}.png",
    )
    print("Done.")


if __name__ == "__main__":
    main()
