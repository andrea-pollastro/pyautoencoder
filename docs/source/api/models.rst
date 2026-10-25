Autoencoder Models
===================

:class:`~pyautoencoder.vanilla.AE` and :class:`~pyautoencoder.variational.VAE`
inherit :meth:`~pyautoencoder._base.BaseAutoencoder.build`,
:meth:`~pyautoencoder._base.BaseAutoencoder.encode` and
:meth:`~pyautoencoder._base.BaseAutoencoder.decode` from
:class:`~pyautoencoder._base.BaseAutoencoder` (see :doc:`base`).

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
   :no-index:

.. autoclass:: pyautoencoder.vanilla.AEDecodeOutput
   :members:
   :no-index:

.. autoclass:: pyautoencoder.vanilla.AEOutput
   :members:
   :no-index:

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
   :no-index:

.. autoclass:: pyautoencoder.variational.VAEDecodeOutput
   :members:
   :no-index:

.. autoclass:: pyautoencoder.variational.VAEOutput
   :members:
   :no-index:

Adaptive Group Variational Autoencoder
---------------------------------------

.. autoclass:: pyautoencoder.variational.AdaGVAE
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: _encode, _decode
   :special-members: __init__
