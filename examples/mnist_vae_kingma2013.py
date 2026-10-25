"""
Reproduce the MNIST VAE experiment from Kingma & Welling (2013), Fig. 2.

We train VAEs with different latent dimensionalities N_z and track the ELBO
as a function of the number of training samples seen.
The setup follows the original paper:

- One hidden layer MLPs with Tanh activations in encoder/decoder
- Hidden size H = 500
- Mini-batch size M = 100
- Adagrad, learning rate selected from {0.01, 0.02, 0.1} (we use 0.02 here)
- All parameters initialized from N(0, 0.01) (variance 0.01, i.e. std 0.1)
- Small weight decay corresponding to a prior N(0, I) on all parameters
- L = 1 Monte Carlo sample for the stochastic latent variable
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.ticker import (
    LogFormatterMathtext,
    LogLocator,
    NullFormatter,
)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from pyautoencoder.variational import VAE


# ---------------- Configuration ---------------- #
LATENT_DIMS: List[int] = [3, 5, 10, 20, 200]  # N_z values (paper)
HIDDEN_SIZE: int = 500                        # hidden layer size (encoder/decoder, MNIST)
BATCH_SIZE: int = 100                         # M = 100 (paper)
LEARNING_RATE: float = 0.02                   # chosen from {0.01, 0.02, 0.1} (paper)
MC_SAMPLES: int = 1                           # L = 1 (paper)
TARGET_TRAIN_SAMPLES: int = int(1e7)          # stop after this many training samples
EVAL_EVERY_SAMPLES: int = int(1e5)            # evaluate and log every this many samples
USE_STOCHASTIC_BINARIZATION: bool = False     # binarize x ~ Bernoulli(p = p_x) if True

DEVICE = torch.device("cpu")
SEED: int = 1926


# ---------------- Utilities ------------------- #
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def init_params_paper(model: nn.Module) -> None:
    # Paper, Sec. 5: "All parameters, both variational and generative, were initialized
    # by random sampling from N(0, 0.01)". As in N(0, I), the second argument is the
    # variance: std = 0.1, for weights and biases alike.
    for param in model.parameters():
        param.normal_(mean=0.0, std=0.1)


# ---------------- Data ----------------------- #
def make_dataloaders(
    batch_size: int,
    use_stochastic_binarization: bool = USE_STOCHASTIC_BINARIZATION,
) -> Tuple[DataLoader, DataLoader]:

    tfms: List[transforms.Compose | transforms.ToTensor] = [transforms.ToTensor()]

    if use_stochastic_binarization:
        class BernoulliBinarize:
            def __call__(self, x: torch.Tensor) -> torch.Tensor:
                # Sample x ~ Bernoulli(p = x); assumes x in [0, 1].
                return torch.bernoulli(x)

        tfms.append(BernoulliBinarize()) # type: ignore

    transform = transforms.Compose(tfms)

    train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_dataset  = datasets.MNIST("./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, test_loader


# ---------------- Model ---------------------- #
def make_vae(latent_dim: int, hidden: int = HIDDEN_SIZE) -> VAE:
    encoder = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, hidden),
        nn.Tanh(),
    )

    decoder = nn.Sequential(
        nn.Linear(latent_dim, hidden),
        nn.Tanh(),
        nn.Linear(hidden, 28 * 28),
        nn.Unflatten(-1, (1, 28, 28)),  # keep last layer linear: the Bernoulli likelihood expects logits
    )

    model = VAE(encoder=encoder, decoder=decoder, latent_dim=latent_dim)
    model.build(input_sample=torch.randn(1, 1, 28, 28))
    init_params_paper(model)
    return model

# ---------------- Evaluation ----------------- #
@torch.no_grad()
def average_elbo(dataloader: DataLoader, model: VAE) -> float:
    model.eval()  # forward still samples z ~ q(z|x), so this is an ELBO estimate
    total_elbo = torch.zeros((), device=DEVICE)
    n = 0

    for x, _ in dataloader:
        x = x.to(DEVICE)
        out = model(x, S=MC_SAMPLES)
        loss_info = model.compute_loss(x, out, beta=1, likelihood='bernoulli')
        elbo_batch = loss_info.diagnostics['elbo']
        batch_size = x.size(0)
        total_elbo += elbo_batch * batch_size
        n += batch_size

    return total_elbo.item() / n


# ---------------- Training ------------------- #
def train_one_setting(
    latent_dim: int,
    train_loader: DataLoader,
    test_loader: DataLoader,
) -> Tuple[VAE, List[Dict[str, float]]]:
    model = make_vae(latent_dim).to(DEVICE)

    # Paper, Sec. 5: "a small weight decay term corresponding to a prior p(theta) = N(0, I)",
    # and all parameters "were jointly stochastically optimized using the MAP criterion",
    # so the prior is applied to encoder and decoder alike. The loss is a per-datapoint mean
    # (the dataset objective divided by N), so the prior contributes -||params||^2 / (2N):
    # a weight decay of 1/N.
    optimizer = torch.optim.Adagrad(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1.0 / len(train_loader.dataset),
    )

    logs: List[Dict[str, float]] = []
    samples_seen = 0
    next_eval = EVAL_EVERY_SAMPLES

    start_time = time.time()
    while samples_seen < TARGET_TRAIN_SAMPLES:
        for x, _ in train_loader:
            x = x.to(DEVICE)
            model.train()
            optimizer.zero_grad()

            out = model(x, S=MC_SAMPLES)
            loss_info = model.compute_loss(x, out, beta=1, likelihood='bernoulli')
            loss_info.objective.backward()
            optimizer.step()

            batch_size = x.size(0)
            samples_seen += batch_size

            if samples_seen >= next_eval:
                train_elbo = average_elbo(train_loader, model)
                test_elbo = average_elbo(test_loader, model)
                logs.append(
                    {
                        "samples": float(samples_seen),
                        "train_elbo": float(train_elbo),
                        "test_elbo": float(test_elbo),
                    }
                )
                elapsed = time.time() - start_time
                print(
                    f"N_z={latent_dim:3d} | "
                    f"samples={samples_seen:>9d} | "
                    f"ELBO_train={train_elbo:.2f}, "
                    f"ELBO_test={test_elbo:.2f} | "
                    f"(elapsed {elapsed:.1f}s)"
                )
                next_eval += EVAL_EVERY_SAMPLES

            if samples_seen >= TARGET_TRAIN_SAMPLES:
                break

    return model, logs


# ---------------- Plotting ------------------- #
def plot_elbo_curves(all_logs: Dict[int, List[Dict[str, float]]]) -> Path:
    num_settings = len(all_logs)
    fig, axes = plt.subplots(1, num_settings, figsize=(3.0 * num_settings, 3.0))

    # Ensure axes is iterable even when num_settings == 1
    if num_settings == 1:
        axes = np.array([axes])

    latent_dims_sorted = sorted(all_logs.keys())

    for i, nz in enumerate(latent_dims_sorted):
        ax = axes[i]
        xs = [entry["samples"] for entry in all_logs[nz]]
        ys_train = [entry["train_elbo"] for entry in all_logs[nz]]
        ys_test = [entry["test_elbo"] for entry in all_logs[nz]]

        ax.plot(xs, ys_train, label="AEVB (train)", color="r")
        ax.plot(xs, ys_test, linestyle="--", label="AEVB (test)", color="r")

        ax.set_xscale("log")
        ax.set_ylim(-150, -95)
        ax.set_title(f"MNIST, $N_z = {nz}$")

        if i == 0:
            ax.set_xlabel("# training samples evaluated")
            ax.set_ylabel(r"$\mathcal{L}$ (ELBO)")

        # Log-scale formatting on x-axis
        ax.xaxis.set_major_locator(LogLocator(base=10))
        ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
        ax.xaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10)))
        ax.xaxis.set_minor_formatter(NullFormatter())

    axes[0].legend(loc="lower right")
    plt.tight_layout()

    out_path = Path("vae_mnist_fig2_repro.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure -> {out_path}")
    return out_path


# ---------------- Main ---------------------- #
def main() -> None:
    print("=== Reproducing Kingma & Welling (2013) Fig. 2 on MNIST ===")
    print(f"Using device: {DEVICE}")

    set_seed(SEED)
    train_loader, test_loader = make_dataloaders(BATCH_SIZE)

    all_logs: Dict[int, List[Dict[str, float]]] = {}
    for nz in LATENT_DIMS:
        print(f"\n--- Training VAE with N_z = {nz} ---")
        _, logs = train_one_setting(nz, train_loader, test_loader)
        all_logs[nz] = logs

    plot_elbo_curves(all_logs)
    print("Done.")

if __name__ == "__main__":
    main()
