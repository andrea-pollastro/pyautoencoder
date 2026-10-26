Autoencoder Models
===================

:class:`~pyautoencoder.vanilla.AE` and :class:`~pyautoencoder.variational.VAE`
share ``build``, ``encode`` and ``decode``;
:meth:`VAE.encode <pyautoencoder.variational.VAE.encode>` returns the posterior
mean by default. Build your model by passing your own encoder and decoder to
these classes: the base classes they share are internal and not meant to be
subclassed.

Vanilla Autoencoder
-------------------

.. autoclass:: pyautoencoder.vanilla.AE
   :members:
   :undoc-members:
   :inherited-members: Module
   :exclude-members: _encode, _decode, training
   :special-members: __init__

.. rubric:: Data Structures

.. autoclass:: pyautoencoder.vanilla.AEEncodeOutput
   :members:

.. autoclass:: pyautoencoder.vanilla.AEDecodeOutput
   :members:

.. autoclass:: pyautoencoder.vanilla.AEOutput
   :members:

Variational Autoencoder
-----------------------

.. autoclass:: pyautoencoder.variational.VAE
   :members:
   :undoc-members:
   :inherited-members: Module
   :exclude-members: _encode, _decode, training
   :special-members: __init__

.. rubric:: Data Structures

.. autoclass:: pyautoencoder.variational.VAEEncodeOutput
   :members:

.. autoclass:: pyautoencoder.variational.VAEDecodeOutput
   :members:

.. autoclass:: pyautoencoder.variational.VAEOutput
   :members:

Adaptive Group Variational Autoencoder
---------------------------------------

.. autoclass:: pyautoencoder.variational.AdaGVAE
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: _encode, _decode
   :special-members: __init__

.. rubric:: Data Structures

.. autoclass:: pyautoencoder.variational.AdaGVAEOutput
   :members:
