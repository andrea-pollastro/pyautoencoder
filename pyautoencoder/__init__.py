from . import loss, vanilla, variational

try:
    from ._version import version as __version__
except ImportError:  # source checkout that was never built or installed
    __version__ = "0+unknown"

__all__ = [
    "loss",
    "vanilla",
    "variational",
]
