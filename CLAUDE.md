# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`pyautoencoder` is a small PyTorch library (published on PyPI) that wraps user-supplied encoder/decoder `nn.Module`s into ready-to-train autoencoders with matching losses: `vanilla.AE`, `variational.VAE` and `variational.AdaGVAE`. It deliberately has no training loop or framework abstractions. The only runtime dependency is `torch`.

## Commands

```bash
pip install -e ".[dev]"            # package + pytest, sphinx, ruff

pytest                             # full suite (testpaths = test/)
pytest test/variational/test_vae.py                                   # one file
pytest test/variational/test_vae.py::test_vae_build_materializes_lazy_layers  # one test
pytest -k adagvae                  # by keyword

ruff check .                       # lint (CI pins ruff==0.17.2)

# Docs: CI builds with -W, so any Sphinx warning (broken ref, RST error) fails the build
sphinx-build -W --keep-going -b html docs/source docs/build/html
```

- pytest sets `pythonpath = ["."]`, so tests import this checkout, not an installed copy.
- `test/conftest.py` has an autouse fixture that forces grad mode on for each test and restores it afterwards.
- Examples (`examples/*.py`) need `torchvision`, `matplotlib` and `numpy`, which are not package dependencies. They run on CPU and download MNIST to `./data`.

## Compatibility constraints

- Python >= 3.10 and torch >= 2.1.0. CI tests Python 3.10–3.14 against the latest CPU torch, plus Python 3.10 with torch 2.1.0. Don't use torch APIs newer than 2.1 without checking.
- The version comes from git tags via setuptools-scm (`pyautoencoder/_version.py` is generated and gitignored; `__version__` falls back to `"0+unknown"`). Pushing a `v*` tag runs CI and then publishes to PyPI (`.github/workflows/publish.yml`).

## Architecture

### Shared contract (`pyautoencoder/_base/base.py`)

`BaseAutoencoder(nn.Module, ABC)` splits each model into:

- `_encode`, `_decode`, `forward`: gradient-enabled; subclasses implement them and use them for training.
- `encode`, `decode`: public inference wrappers that run under `torch.inference_mode()`. Subclasses may override them to change defaults (see VAE).
- `build(input_sample)`: a no-grad forward pass in eval mode that materializes lazy layers. It restores each submodule's original `training` flag. It must be called before creating the optimizer, but is not needed before `load_state_dict`.
- `compute_loss(x, output, ...) -> LossResult`: the second argument is always named `output`.

All model outputs are `@dataclass(slots=True, repr=False)` subclasses of `ModelOutput`. `repr=False` keeps the base class's tensor-aware `__repr__`, which prints shape and dtype instead of values. AE and VAE each have an `*EncodeOutput`, a `*DecodeOutput` and a full `*Output`; AdaGVAE has only `AdaGVAEOutput` (a pair of `VAEOutput`s).

### Losses (`pyautoencoder/loss/base.py`)

- `LossResult.objective` is the scalar to backprop. `LossResult.diagnostics` holds **detached 0-dim tensors, not floats**, so the loss doesn't force a GPU sync; callers run `.item()` when they log.
- `log_likelihood` is elementwise and raises on any shape mismatch: broadcasting is deliberately rejected. The Gaussian case has unit variance and no normalization constant. For Bernoulli, `x_hat` is **logits**.
- Every model reduces the same way: elementwise log-likelihood → sum over feature dims → mean over the batch.

### VAE (`pyautoencoder/variational/`)

- The encoder must output flat features `[B, F]`. `FullyFactorizedGaussian` maps them to `mu`/`log_var` with `nn.LazyLinear`, which is why `build()` is required.
- Latents always carry a Monte Carlo sample dimension: `z` is `[B, S, D_z]` even when `S=1`, and `x_hat` is `[B, S, ...]`. `_decode` rejects anything else and reshapes to `[B*S, D_z]` before calling the user's decoder.
- The explicit `sample` argument decides between sampling and the posterior mean; train/eval mode has no effect. `forward`/`_encode` default to `sample=True`, while `VAE.encode` defaults to `sample=False` and returns the posterior mean. `sample=False` requires `S=1`.
- `compute_loss` takes the Monte Carlo average of the log-likelihood over `S`, subtracts `beta`·KL(q‖N(0,I)), and returns the negative ELBO.
- `compute_loss` first checks that `x_hat` is `[B, S, *x.shape[1:]]` and that `mu`/`log_var` are `[B, latent_dim]`, and raises otherwise. It never infers `S` from the shapes (any guess is ambiguous), and it broadcasts `x` over `S` only.

### AdaGVAE

`AdaGVAE` is a plain `nn.Module`, not a `BaseAutoencoder`. It wraps an existing `VAE` and takes a pair `(x1, x2)`. In **training mode** it computes per-dimension KL(q1‖q2), uses the midpoint of its min and max as a threshold, and averages the two posteriors on the dimensions below it. It then samples through `sampling_layer.reparametrize`. In **eval mode** no grouping is applied, and it runs each input through `VAE.forward` independently. For single-input inference, use `model.vae.encode` / `model.vae.decode`.

## Docs coupling

- `docs/source/examples/*.rst` pull code from `examples/*.py` with `literalinclude` + `:pyobject:`. Renaming or removing a function in an example breaks the docs build.
- The API pages (`docs/source/api/*.rst`) list classes by hand with `autoclass`. New public classes must be added there, and also to the subpackage's `__init__.py` `__all__`.
- Docstrings use NumPy style (parsed by napoleon), with math written as RST `:math:` / `.. math::`.
- Docstring section headers that are not standard NumPy sections (e.g. "Shape convention") must be listed in `napoleon_custom_sections` in `docs/source/conf.py`. Otherwise they become real headings of the page, and the build does not warn.
