# Reproduce Kingma & Welling (2013) Fig. 2 on MNIST with Nz = 3, 5, 10 (AEVB only)

import math, time, random, os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# ---- Repro configuration ----
LATENTS = [3, 5, 10]       # Nz
HIDDEN = 500               # 1 hidden layer size in both encoder/decoder (MNIST)    [paper]
BATCH_SIZE = 100           # M=100                                                  [paper]
LR = 0.02                  # pick from {0.01, 0.02, 0.1}                            [paper]
WEIGHT_DECAY = 1e-4        # small weight decay ~ N(0, I) prior                     [paper mentions small, value here is typical]
MC_SAMPLES = 1             # L = 1                                                  [paper]
TARGET_TRAIN_SAMPLES = 2_000_000    # stop when this many training samples have been seen
EVAL_EVERY_SAMPLES = 50_000         # evaluate and log every this many samples
USE_STOCHASTIC_BINARIZATION = False # set True to binarize x ~ Bernoulli(p=px)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 1926
torch.manual_seed(SEED); random.seed(SEED)

# ---- Import your library exactly as in your snippet ----
from pyautoencoder import VAE, VAELoss  # if your package is named differently, adjust here

# ---- Data ----
base_tfms = [transforms.ToTensor()]  # pixels in [0,1]
if USE_STOCHASTIC_BINARIZATION:
    class BernoulliBinarize(object):
        def __call__(self, x):
            return torch.bernoulli(x)
    base_tfms.append(BernoulliBinarize())

transform = transforms.Compose(base_tfms)
train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_ds  = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

# ---- Utils ----
def init_weights_small_normal(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.01)  # N(0, 0.01) as in paper
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def make_vae(latent_dim, hidden=HIDDEN):
    # Paper uses single hidden layer MLPs; Tanh was common in early VAEs.
    encoder = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, hidden),
        nn.Tanh(),
    )
    decoder = nn.Sequential(
        nn.Linear(latent_dim, hidden),
        nn.Tanh(),
        nn.Linear(hidden, 28*28),
        nn.Unflatten(-1, (1, 28, 28))  # keep last layer linear; VAELoss(bernoulli) should handle logits
    )
    model = VAE(encoder=encoder, decoder=decoder, latent_dim=latent_dim)
    model.build(input_sample=torch.randn(1, 1, 28, 28))
    model.apply(init_weights_small_normal)
    return model

# Assumes VAELoss.total is a loss to MINIMIZE (negative ELBO on average per datapoint).
# If your implementation differs, flip the sign in the two 'elbo_batch' lines below.
loss_fn = VAELoss(likelihood='bernoulli')

@torch.no_grad()
def average_elbo(dataloader, model):
    model.eval()
    total_elbo = 0.0
    n = 0
    for x, _ in dataloader:
        x = x.to(DEVICE)
        out = model(x, S=MC_SAMPLES)  # if your API has a samples arg; else remove
        info = loss_fn(x, out)
        elbo_batch = -info.total.item()  # ELBO ≈ -loss
        bsz = x.size(0)
        total_elbo += elbo_batch * bsz
        n += bsz
    return total_elbo / n

def train_one_setting(latent_dim):
    model = make_vae(latent_dim).to(DEVICE)
    opt = torch.optim.Adagrad(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    logs = []  # list of dicts: {'samples': int, 'train_elbo': float, 'test_elbo': float}
    samples_seen = 0
    next_eval = EVAL_EVERY_SAMPLES

    t0 = time.time()
    while samples_seen < TARGET_TRAIN_SAMPLES:
        for x, _ in train_loader:
            x = x.to(DEVICE)
            model.train()
            opt.zero_grad()
            out = model(x, S=MC_SAMPLES)
            info = loss_fn(x, out)
            info.total.backward()
            opt.step()

            samples_seen += x.size(0)

            if samples_seen >= next_eval:
                tr_elbo = average_elbo(train_loader, model)  # full-train avg; for speed you may sub-sample
                te_elbo = average_elbo(test_loader,  model)
                logs.append({'samples': samples_seen, 'train_elbo': tr_elbo, 'test_elbo': te_elbo})
                print(f"Nz={latent_dim:2d}  samples={samples_seen:>8d}  L_train={tr_elbo:.2f}  L_test={te_elbo:.2f}  (+{time.time()-t0:.1f}s)")
                next_eval += EVAL_EVERY_SAMPLES

            if samples_seen >= TARGET_TRAIN_SAMPLES:
                break
    return model, logs

all_logs = {}
for nz in LATENTS:
    print(f"\n=== Training VAE with Nz={nz} ===")
    _, logs = train_one_setting(nz)
    all_logs[nz] = logs

# ---- Plot (matches the style of Fig. 2) ----
plt.figure(figsize=(8, 6))
for nz in LATENTS:
    xs = [d['samples'] / 1e6 for d in all_logs[nz]]  # in millions of samples
    ys_tr = [d['train_elbo'] for d in all_logs[nz]]
    ys_te = [d['test_elbo']  for d in all_logs[nz]]
    plt.plot(xs, ys_tr, label=f"AEVB (train), Nz={nz}")
    plt.plot(xs, ys_te, linestyle='--', label=f"AEVB (test), Nz={nz}")

plt.xlabel("# Training samples evaluated (millions)")
plt.ylabel("L (average variational lower bound per datapoint)")
plt.title("MNIST VAE – AEVB only (reproduction of Fig. 2 for Nz=3,5,10)")
plt.legend()
plt.tight_layout()
plt.savefig("vae_mnist_fig2_repro_Nz_3_5_10.png", dpi=200)
print("Saved figure -> vae_mnist_fig2_repro_Nz_3_5_10.png")
