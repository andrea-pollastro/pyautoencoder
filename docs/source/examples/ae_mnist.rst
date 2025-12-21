.. _mnist_ae_example:

MNIST Vanilla Autoencoder Example
=================================

This example demonstrates how to train a fully connected autoencoder on
the MNIST dataset using :mod:`pyautoencoder`.

Configuration and Utilities
---------------------------

We begin by importing the required libraries, defining the main
hyperparameters, and setting up deterministic random seeds to ensure
reproducibility.

.. literalinclude:: ../../../examples/mnist_ae.py
   :language: python
   :linenos:
   :lines: 1-33

Key elements in this section include:

* :data:`LATENT_DIM`, :data:`NUM_EPOCHS`, :data:`BATCH_SIZE` –
  the primary model and training hyperparameters,
* :data:`DEVICE` – automatically selects ``"cuda"`` when available,
* :func:`set_seed` – helper function to provide reproducible runs.


Data Loading
------------

We prepare the MNIST training and test datasets using
:class:`torchvision.datasets.MNIST`. A simple ``ToTensor`` transform is
used to map images to the :math:`[0, 1]` range.

.. literalinclude:: ../../../examples/mnist_ae.py
   :language: python
   :linenos:
   :lines: 37-44

The :func:`make_dataloaders` function returns two
:class:`torch.utils.data.DataLoader` objects that supply batches to the
training and evaluation loops.


Model Definition
----------------

The autoencoder consists of a compact fully connected encoder and
decoder. The encoder flattens the input and maps it into a latent
representation of dimension :data:`LATENT_DIM`; the decoder reconstructs
the image from this latent vector.

.. literalinclude:: ../../../examples/mnist_ae.py
   :language: python
   :linenos:
   :lines: 48-63

Notes:

* The final decoder layer is **linear** and produces logits.
* The model is explicitly built through :meth:`AE.build`, which infers
  required shapes from a representative sample.
* Reconstruction loss is computed via the :meth:`AE.compute_loss` method,
  which uses a Bernoulli likelihood (interpreting the decoder output as logits).


Training Loop
-------------

Training uses standard mini-batch gradient descent with the
:class:`torch.optim.Adam` optimizer. During each iteration, we compute
the reconstruction log-likelihood and related diagnostics.

.. literalinclude:: ../../../examples/mnist_ae.py
   :language: python
   :linenos:
   :lines: 66-101

For each batch:

* The model produces latent codes and reconstructions: ``out = model(x)``.
* :meth:`AE.compute_loss` returns a :class:`~pyautoencoder.loss.LossResult`
  containing:

  - ``objective`` – batch-mean negative log-likelihood (NLL) in nats,
  - ``diagnostics['log_likelihood']`` – batch-mean log-likelihood (negative of objective).

* The loss is backpropagated through the entire model.
* These quantities are accumulated over the epoch and reported in the log.


Visualizing Reconstructions
---------------------------

After training completes, we visualize a few randomly chosen test images
alongside their reconstructions. The decoder outputs logits, so we apply
a sigmoid to convert them to pixel intensities suitable for display.

.. literalinclude:: ../../../examples/mnist_ae.py
   :language: python
   :linenos:
   :lines: 105-145


Putting It All Together
-----------------------

The :func:`main` function orchestrates the full example: seeding,
dataset preparation, model creation, training, and saving a figure with
test-set reconstructions.

.. literalinclude:: ../../../examples/mnist_ae.py
   :language: python
   :linenos:
   :lines: 149-165

The script can also be executed directly:

.. literalinclude:: ../../../examples/mnist_ae.py
   :language: python
   :linenos:
   :lines: 168-169


Full Example Script
-------------------

For completeness, the entire example script is shown below:

.. literalinclude:: ../../../examples/mnist_ae.py
   :language: python
   :linenos:
   :caption: mnist_ae.py
