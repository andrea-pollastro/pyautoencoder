"""Internal base classes shared by the models of this package.

:class:`BaseAutoencoder` defines the contract of :class:`~pyautoencoder.vanilla.AE`
and :class:`~pyautoencoder.variational.VAE`, and :class:`ModelOutput` is the
base of their output dataclasses.

Warning
-------
These classes are not part of the public API: they are not meant to be
imported or subclassed by users, and their contract may change between
releases. To build a model, pass your own encoder and decoder to
:class:`~pyautoencoder.vanilla.AE` or :class:`~pyautoencoder.variational.VAE`.
"""

from .base import (
    BaseAutoencoder,
    ModelOutput,
)

__all__ = [
    'BaseAutoencoder',
    'ModelOutput',
]
