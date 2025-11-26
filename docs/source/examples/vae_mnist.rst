.. _mnist_vae_fig2_example:

Reproducing Kingma & Welling (2013), Fig. 2 (MNIST VAE)
=======================================================

This example reproduces Figure 2 from the paper on Auto-Encoding Variational Bayes
(Kingma & Welling (2013)). The experiment trains Variational Autoencoders (VAEs) with
different latent dimensionalities :math:`N_z` on MNIST and tracks the
Evidence Lower Bound (ELBO) as a function of the number of training
samples processed.

The setup follows the configuration in the paper:

* one-hidden-layer MLP encoder and decoder with Tanh activations;
* hidden size :math:`H = 500`;
* mini-batch size :math:`M = 100`;
* learning rate selected from :math:`\{0.01, 0.02, 0.1\}` (here: ``0.02``);
* small weight decay (approximate :math:`\mathcal{N}(0, I)` prior on weights);
* :math:`L = 1` Monte Carlo sample for the latent variable estimator.


Configuration
-------------

We begin with imports, global settings, and the reproducibility utilities.
This includes the list of latent dimensions used for :math:`N_z`,
hyperparameters, and device configuration.

.. literalinclude:: ../../../examples/mnist_vae_kingma2013.py
   :language: python
   :linenos:
   :lines: 1-65

Key parameters:

* :data:`LATENT_DIMS` – list of latent sizes :math:`N_z` evaluated.
* :data:`HIDDEN_SIZE` – hidden layer size in encoder and decoder.
* :data:`MC_SAMPLES` – number of Monte Carlo samples (:math:`L=1` in the paper).
* :func:`set_seed` – deterministic seeding for reproducibility.


Data Loading
------------

The MNIST dataset is loaded through :mod:`torchvision.datasets`.
Optionally, inputs may be stochastically binarized as in the original
AEVB experiments.

.. literalinclude:: ../../../examples/mnist_vae_kingma2013.py
   :language: python
   :linenos:
   :lines: 69-91

The :func:`make_dataloaders` function returns training and test dataloaders
with appropriate preprocessing and optional binarization.


Model Definition
----------------

Each VAE consists of:

* a single-hidden-layer encoder mapping :math:`x \mapsto (\mu, \log \sigma^2)`,
* a single-hidden-layer decoder mapping :math:`z \mapsto \hat{x}`,
* Tanh nonlinearities,
* linear output logits interpreted by :class:`~pyautoencoder.loss.VAELoss`
  under a Bernoulli likelihood.

Weights are initialized with a small Gaussian as described in the paper.

.. literalinclude:: ../../../examples/mnist_vae_kingma2013.py
   :language: python
   :linenos:
   :lines: 95-114

The model is built explicitly via :meth:`VAE.build`, which infers
dimension-dependent components from a representative input sample.


ELBO Evaluation
---------------

We evaluate the ELBO over a dataloader by negating the average
negative-ELBO returned by :class:`VAELoss`.

.. literalinclude:: ../../../examples/mnist_vae_kingma2013.py
   :language: python
   :linenos:
   :lines: 117-132

This routine is used during training to record both training and test
ELBO values.


Training Loop
-------------

For each latent dimension :math:`N_z`, the VAE is trained until a fixed
number of training samples have been processed
(:data:`TARGET_TRAIN_SAMPLES`). Periodic evaluations of the ELBO on both
train and test sets are logged.

.. literalinclude:: ../../../examples/mnist_vae_kingma2013.py
   :language: python
   :linenos:
   :lines: 136-189

Each evaluation step records:

* number of processed samples,
* average ELBO on the training set,
* average ELBO on the test set.


Plotting the Results
--------------------

After all training runs complete, the ELBO curves are plotted as a
function of the number of processed samples (log-scaled), following the
appearance of Fig. 2 from Kingma & Welling (2013).

.. literalinclude:: ../../../examples/mnist_vae_kingma2013.py
   :language: python
   :linenos:
   :lines: 193-233

The figure is saved as ``vae_mnist_fig2_repro.png``.


Entry Point
-----------

The :func:`main` function orchestrates the full experiment: seeding,
data loading, training VAEs for each :math:`N_z`, logging results, and
plotting the final curves.

.. literalinclude:: ../../../examples/mnist_vae_kingma2013.py
   :language: python
   :linenos:
   :lines: 237-251

The script can be executed directly:

.. literalinclude:: ../../../examples/mnist_vae_kingma2013.py
   :language: python
   :linenos:
   :lines: 253-257


Full Example Script
-------------------

For convenience, here is the complete script in one block:

.. literalinclude:: ../../../examples/mnist_vae_kingma2013.py
   :language: python
   :linenos:
   :caption: mnist_vae_kingma2013.py
