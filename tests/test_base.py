"""Test base autoencoder functionality."""
import torch
import pytest
from pyautoencoder.models.base import BaseAutoencoder, BaseVariationalAutoencoder

def test_base_autoencoder_abstract():
    """Test that BaseAutoencoder cannot be instantiated."""
    with pytest.raises(TypeError):
        BaseAutoencoder()

def test_base_vae_abstract():
    """Test that BaseVariationalAutoencoder cannot be instantiated."""
    with pytest.raises(TypeError):
        BaseVariationalAutoencoder()

def test_inheritance():
    """Test proper inheritance relationship."""
    assert issubclass(BaseVariationalAutoencoder, BaseAutoencoder)
