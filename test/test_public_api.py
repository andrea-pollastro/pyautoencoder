import importlib

import pytest


@pytest.mark.parametrize(
    "package", ["pyautoencoder.loss", "pyautoencoder.vanilla", "pyautoencoder.variational"]
)
def test_package_exports_every_name_in_all(package):
    module = importlib.import_module(package)
    for name in module.__all__:
        assert hasattr(module, name), f"{package}.{name}"


@pytest.mark.parametrize(
    "old_path",
    ["pyautoencoder.loss.base", "pyautoencoder.vanilla.autoencoder", "pyautoencoder.variational.vae"],
)
def test_implementation_modules_are_private(old_path):
    # Public names are imported from the packages, never from the implementation modules
    with pytest.raises(ImportError):
        importlib.import_module(old_path)
