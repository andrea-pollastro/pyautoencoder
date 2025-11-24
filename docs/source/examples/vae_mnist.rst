MNIST Variational Autoencoder (VAE) Example
===========================================

This example trains a Variational Autoencoder on MNIST following the structure
of the original Kingma & Welling (2013) paper.

The model uses the :class:`pyautoencoder.variational.VAE` class and the
corresponding :class:`pyautoencoder.loss.VAELoss`.

Code
----

.. literalinclude:: ../../../examples/mnist_vae_kingma2013.py
   :language: python
   :linenos:
   :caption: mnist_vae_kingma2013.py
