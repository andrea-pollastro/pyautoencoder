"""Internal base classes and utilities for autoencoders.

This module provides the abstract :class:`BaseAutoencoder` interface,
the build-guard mixin :class:`BuildGuardMixin`, and the output container
:class:`ModelOutput`. These form the foundation for all autoencoder
implementations in this package.

Warning
-------
This is an internal API. Most users should work with :class:`vanilla.AE` or
:class:`variational.VAE` instead of using these base classes directly.
"""

from .base import (
    BaseAutoencoder,
    ModelOutput,
    NotBuiltError,
)

__all__ = [
    'BaseAutoencoder',
    'ModelOutput',
    'NotBuiltError',
]
