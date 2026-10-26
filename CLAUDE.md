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

# Docs: CI builds with -W, so any Sphinx warning (RST error, missing include pattern) fails the build
sphinx-build -W --keep-going -b html docs/source docs/build/html
# Unresolved Python references (:class:, :meth:) only warn in nitpicky mode; list them with:
sphinx-build -n -E -b html docs/source docs/build/html
```

- pytest sets `pythonpath = ["."]`, so tests import this checkout, not an installed copy.
- `test/conftest.py` has an autouse fixture that forces grad mode on for each test and restores it afterwards.
- Examples (`examples/*.py`) need `torchvision`, `matplotlib` and `numpy`, which are not package dependencies. They run on CPU and download MNIST to `./data`.

## Compatibility constraints

- Python >= 3.10 and torch >= 2.1.0. CI tests Python 3.10–3.14 against the latest CPU torch, plus Python 3.10 with torch 2.1.0. Don't use torch APIs newer than 2.1 without checking.
- The version comes from git tags via setuptools-scm (`pyautoencoder/_version.py` is generated and gitignored; `__version__` falls back to `"0+unknown"`). Pushing a `v*` tag runs CI and then publishes to PyPI (`.github/workflows/publish.yml`). Every push to `main` runs CI and rebuilds and deploys the docs to GitHub Pages (`.github/workflows/docs.yml`).

## Architecture

### Public API

Users import from the subpackages `pyautoencoder.vanilla`, `pyautoencoder.variational` and `pyautoencoder.loss` (the names in their `__init__.py` `__all__`). Implementation modules are private: `vanilla/_autoencoder.py`, `variational/_vae.py`, `loss/_base.py` and the `_base` package. Docs, README and examples never import from them, and tests import public names from the packages too. The only public submodule is `variational/stochastic_layers.py` (see VAE).

The base classes (`BaseAutoencoder`, `ModelOutput`) are private and not meant to be subclassed by users: users extend the library by passing their own encoder and decoder, not by inheriting. Keeping them private leaves their contract free to change; making them public later would not break anything, the opposite would.

### Shared contract (`pyautoencoder/_base/base.py`)

`BaseAutoencoder(nn.Module, ABC)` splits each model into:

- `_encode`, `_decode`, `forward`: gradient-enabled; subclasses implement them and use them for training.
- `encode`, `decode`: public inference wrappers that run under `torch.no_grad()`. Not `inference_mode`: its outputs could not be used later in autograd (e.g. frozen-encoder features). They don't change the train/eval mode. Subclasses may override them to change defaults (see VAE).
- `build(input_sample)`: a no-grad forward pass in eval mode that materializes lazy layers. It restores each submodule's original `training` flag. It must be called before creating the optimizer, but is not needed before `load_state_dict`.
- `compute_loss(x, output, ...) -> LossResult`: the second argument is always named `output`.

All model outputs are `@dataclass(slots=True, repr=False)` subclasses of `ModelOutput`. `repr=False` keeps the base class's tensor-aware `__repr__`, which prints shape and dtype instead of values. AE and VAE each have an `*EncodeOutput`, a `*DecodeOutput` and a full `*Output`; AdaGVAE has only `AdaGVAEOutput` (a pair of `VAEOutput`s).

### Losses (`pyautoencoder/loss/_base.py`)

- `LossResult.objective` is the scalar to backprop. `LossResult.diagnostics` holds **detached 0-dim tensors, not floats**, so the loss doesn't force a GPU sync; callers run `.item()` when they log.
- `log_likelihood` is elementwise and raises on any shape mismatch: broadcasting is deliberately rejected. The Gaussian case has unit variance and no normalization constant. For Bernoulli, `x_hat` is **logits**.
- Every model reduces the same way: elementwise log-likelihood → sum over feature dims → mean over the batch. The VAE takes the mean over `S` before the batch mean; AdaGVAE sums the two VAE objectives of the pair.

### VAE (`pyautoencoder/variational/`)

- The encoder must output flat features `[B, F]`. `FullyFactorizedGaussian` maps them to `mu`/`log_var` with `nn.LazyLinear`, which is why `build()` is required.
- `FullyFactorizedGaussian` must be importable only as `from pyautoencoder.variational.stochastic_layers import FullyFactorizedGaussian` (at runtime it is `vae.sampling_layer`). It is deliberately not in `pyautoencoder.variational` or its `__all__`, and `_vae.py` imports the module, not the class, so it is not a name of `_vae` either.
- Latents always carry a Monte Carlo sample dimension: `z` is `[B, S, D_z]` even when `S=1`, and `x_hat` is `[B, S, ...]`. `_decode` rejects anything else and reshapes to `[B*S, D_z]` before calling the user's decoder.
- The explicit `sample` argument decides between sampling and the posterior mean; train/eval mode has no effect. `forward`/`_encode` default to `sample=True`, while `VAE.encode` defaults to `sample=False` and returns the posterior mean. `sample=False` requires `S=1`.
- `compute_loss` first checks that `x_hat` is `[B, S, *x.shape[1:]]` and that `mu`/`log_var` are `[B, latent_dim]`, and raises otherwise. It never infers `S` from the shapes (any guess is ambiguous), and it broadcasts `x` over `S` only. It then takes the Monte Carlo average of the log-likelihood over `S`, subtracts `beta`·KL(q‖N(0,I)), and returns the negative of that as `objective`. The `"elbo"` diagnostic is this `beta`-weighted value, so it is the ELBO only when `beta=1`.

### AdaGVAE

`AdaGVAE` is a plain `nn.Module`, not a `BaseAutoencoder`. It wraps an existing `VAE` and takes a pair `(x1, x2)`. In **training mode** it computes per-dimension KL(q1‖q2), uses the midpoint of its min and max as a threshold, and averages the two posteriors on the dimensions below it (mean of the means and of the variances, not of the log-variances). It then samples through `sampling_layer.reparametrize`. In **eval mode** no grouping is applied, and it runs each input through `VAE.forward` independently. For single-input inference, use `model.vae.encode` / `model.vae.decode`.

`AdaGVAE.build` takes a single input batch, not a pair, and delegates to `VAE.build`. `AdaGVAE.compute_loss` sums `VAE.compute_loss` over the two outputs, so it inherits its shape checks.

## Docs coupling

- `docs/source/examples/*.rst` pull code from `examples/*.py` with `literalinclude`: functions by name (`:pyobject:`), plus the `# ---------------- Data` comment (`:end-before:`) and `if __name__ == "__main__":` (`:start-at:`). Renaming or removing any of these breaks the docs build.
- `docs/source/api/models.rst` and `losses.rst` list classes and functions by hand (`autoclass`, `autofunction`); `stochastic.rst` uses `automodule`. The private `_base` package has no API page. New public classes must be added there, and also to the subpackage's `__init__.py` `__all__`.
- Output and loss dataclasses document their fields in the docstring's `Attributes` section. Don't add `:undoc-members:` to their `autoclass` (fields would show twice) or `:no-index:` (the class could no longer be linked).
- Cross-reference with the full public path, e.g. ``:meth:`~pyautoencoder.vanilla.AE.compute_loss` ``; short names like `AE.compute_loss` don't resolve in `.rst` pages. AE and VAE show their inherited methods (`build`, `encode`, `decode`) through `:inherited-members: Module`, so link them on the public class (e.g. `pyautoencoder.variational.VAE.build`), never on `_base`.
- intersphinx is deliberately not configured, so torch types (`torch.Tensor`, `nn.Module`) always appear as unresolved in the nitpicky output: ignore them.
- Docstrings use NumPy style (parsed by napoleon), with math written as RST `:math:` / `.. math::`.
- Docstring section headers that are not standard NumPy sections (e.g. "Shape convention") must be listed in `napoleon_custom_sections` in `docs/source/conf.py`. Otherwise they become real headings of the page, and the build does not warn.
