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
* Adagrad, learning rate selected from :math:`\{0.01, 0.02, 0.1\}` (here: ``0.02``);
* all parameters initialized from :math:`\mathcal{N}(0, 0.01)` (variance
  :math:`0.01`, i.e. standard deviation :math:`0.1`);
* small weight decay corresponding to a prior :math:`\mathcal{N}(0, I)` on all
  parameters, i.e. a weight decay of :math:`1/N` with :math:`N` training points,
  since the loss is a per-datapoint mean;
* :math:`L = 1` Monte Carlo sample for the latent variable estimator.


Configuration
-------------

We begin with imports, global settings, and the reproducibility utilities.
This includes the list of latent dimensions used for :math:`N_z`,
hyperparameters, and device configuration.

.. literalinclude:: ../../../examples/mnist_vae_kingma2013.py
   :language: python
   :linenos:
   :end-before: # ---------------- Data

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
   :pyobject: make_dataloaders

The :func:`make_dataloaders` function returns training and test dataloaders
with appropriate preprocessing and optional binarization.


Model Definition
----------------

Each VAE consists of:

* a single-hidden-layer encoder mapping :math:`x \mapsto (\mu, \log \sigma^2)`,
* a single-hidden-layer decoder mapping :math:`z \mapsto \hat{x}`,
* Tanh nonlinearities,
* linear output logits interpreted by :meth:`VAE.compute_loss`
  under a Bernoulli likelihood.

All parameters (weights and biases) are initialized from :math:`\mathcal{N}(0, 0.01)`
as described in the paper, reading the second argument as the variance.

.. literalinclude:: ../../../examples/mnist_vae_kingma2013.py
   :language: python
   :linenos:
   :pyobject: make_vae

The model is built explicitly via :meth:`VAE.build`, which infers
dimension-dependent components from a representative input sample.


ELBO Evaluation
---------------

We evaluate the ELBO over a dataloader by summing the batch-wise ELBO
diagnostics returned by :meth:`VAE.compute_loss`.

The forward pass samples :math:`z \sim q(z \mid x)` also in eval mode, so the
returned ``elbo`` is a Monte Carlo estimate of the ELBO on the evaluated data.

.. literalinclude:: ../../../examples/mnist_vae_kingma2013.py
   :language: python
   :linenos:
   :pyobject: average_elbo

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
   :pyobject: train_one_setting

Each training step:

* Draws :math:`S` Monte Carlo samples from :math:`q(z \mid x)`,
* Computes the negative ELBO via :meth:`VAE.compute_loss` with ``likelihood='bernoulli'``,
* Backpropagates gradients through encoder and decoder,
* Records evaluation metrics periodically.

The ELBO diagnostics contain:

* ``elbo`` – Evidence Lower Bound value,
* ``log_likelihood`` – batch-mean reconstruction term,
* ``kl_divergence`` – batch-mean KL divergence.


Plotting the Results
--------------------

After all training runs complete, the ELBO curves are plotted as a
function of the number of processed samples (log-scaled), following the
appearance of Fig. 2 from Kingma & Welling (2013).

.. literalinclude:: ../../../examples/mnist_vae_kingma2013.py
   :language: python
   :linenos:
   :pyobject: plot_elbo_curves

The figure is saved as ``vae_mnist_fig2_repro.png``.


Entry Point
-----------

The :func:`main` function orchestrates the full experiment: seeding,
data loading, training VAEs for each :math:`N_z`, logging results, and
plotting the final curves.

.. literalinclude:: ../../../examples/mnist_vae_kingma2013.py
   :language: python
   :linenos:
   :pyobject: main

The script can be executed directly:

.. literalinclude:: ../../../examples/mnist_vae_kingma2013.py
   :language: python
   :linenos:
   :start-at: if __name__ == "__main__":


Full Example Script
-------------------

For convenience, here is the complete script in one block:

.. literalinclude:: ../../../examples/mnist_vae_kingma2013.py
   :language: python
   :linenos:
   :caption: mnist_vae_kingma2013.py
