Autoencoder Models
===================

:class:`~pyautoencoder.vanilla.AE` and :class:`~pyautoencoder.variational.VAE`
inherit :meth:`~pyautoencoder._base.BaseAutoencoder.build` and
:meth:`~pyautoencoder._base.BaseAutoencoder.decode` from
:class:`~pyautoencoder._base.BaseAutoencoder` (see :doc:`base`).
:class:`~pyautoencoder.vanilla.AE` also inherits
:meth:`~pyautoencoder._base.BaseAutoencoder.encode`, while
:class:`~pyautoencoder.variational.VAE` overrides it to return the posterior
mean by default (see :meth:`VAE.encode <pyautoencoder.variational.VAE.encode>`).

Vanilla Autoencoder
-------------------

.. autoclass:: pyautoencoder.vanilla.AE
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: _encode, _decode
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
   :show-inheritance:
   :exclude-members: _encode, _decode
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
