import pytest
import torch
import torch.nn as nn

from pyautoencoder._base.base import NotBuiltError
from pyautoencoder.vanilla.autoencoder import (
    AE,
    AEEncodeOutput,
    AEDecodeOutput,
    AEOutput,
)


# ---------- helpers ----------

class SimpleEncoder(nn.Module):
    def __init__(self, in_features: int, latent_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, latent_features, bias=False)
        self.last_grad_enabled: bool | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.last_grad_enabled = torch.is_grad_enabled()
        return self.linear(x)


class SimpleDecoder(nn.Module):
    def __init__(self, latent_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(latent_features, out_features, bias=False)
        self.last_grad_enabled: bool | None = None

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        self.last_grad_enabled = torch.is_grad_enabled()
        return self.linear(z)


# ================= AE =================

def test_ae_requires_build_before_use():
    encoder = SimpleEncoder(in_features=4, latent_features=6)
    decoder = SimpleDecoder(latent_features=6, out_features=4)
    ae = AE(encoder=encoder, decoder=decoder)

    x = torch.randn(2, 4)
    z = torch.randn(2, 6)

    # BaseAutoencoder should have wrapped forward / _encode / _decode
    with pytest.raises(NotBuiltError):
        ae.forward(x)

    with pytest.raises(NotBuiltError):
        ae._encode(x)

    with pytest.raises(NotBuiltError):
        ae._decode(z)

    # encode/decode (inference API) must also respect the guard
    with pytest.raises(NotBuiltError):
        ae.encode(x)

    with pytest.raises(NotBuiltError):
        ae.decode(z)


def test_ae_build_sets_built_flag_and_keeps_modules():
    encoder = SimpleEncoder(in_features=4, latent_features=6)
    decoder = SimpleDecoder(latent_features=6, out_features=4)
    ae = AE(encoder=encoder, decoder=decoder)

    encoder_id = id(ae.encoder)
    decoder_id = id(ae.decoder)

    assert ae.built is False

    x_sample = torch.randn(1, 4)
    ae.build(x_sample)

    # build() is trivial here, but must flip the flag
    assert ae.built is True
    assert ae._built is True

    # build() must not replace encoder/decoder
    assert id(ae.encoder) == encoder_id
    assert id(ae.decoder) == decoder_id


def test_ae_training_forward_encode_decode_shapes_and_types():
    batch_size = 5
    in_features = 8
    latent_features = 3

    encoder = SimpleEncoder(in_features=in_features, latent_features=latent_features)
    decoder = SimpleDecoder(latent_features=latent_features, out_features=in_features)
    ae = AE(encoder=encoder, decoder=decoder)

    x = torch.randn(batch_size, in_features)
    ae.build(x)  # unlock guarded methods

    torch.set_grad_enabled(True)

    # _encode
    enc_out = ae._encode(x)
    assert isinstance(enc_out, AEEncodeOutput)
    assert enc_out.z.shape == (batch_size, latent_features)
    assert enc_out.z.requires_grad is True

    # _decode
    dec_out = ae._decode(enc_out.z)
    assert isinstance(dec_out, AEDecodeOutput)
    assert dec_out.x_hat.shape == (batch_size, in_features)
    assert dec_out.x_hat.requires_grad is True

    # forward
    out = ae.forward(x)
    assert isinstance(out, AEOutput)
    assert out.z.shape == (batch_size, latent_features)
    assert out.x_hat.shape == (batch_size, in_features)
    assert out.z.requires_grad is True
    assert out.x_hat.requires_grad is True

    # encoder/decoder must see grad enabled during training calls
    assert encoder.last_grad_enabled is True
    assert decoder.last_grad_enabled is True


def test_ae_forward_calls_encoder_then_decoder_consistently():
    """Sanity check: forward(x) is equivalent to decode(encode(x).z) in training mode."""
    batch_size = 4
    in_features = 7
    latent_features = 5

    encoder = SimpleEncoder(in_features=in_features, latent_features=latent_features)
    decoder = SimpleDecoder(latent_features=latent_features, out_features=in_features)
    ae = AE(encoder=encoder, decoder=decoder)

    x = torch.randn(batch_size, in_features)
    ae.build(x)

    torch.set_grad_enabled(True)

    # explicit encode/decode via training hooks
    enc_out = ae._encode(x)
    dec_out = ae._decode(enc_out.z)

    # forward
    out = ae.forward(x)

    # z from forward should match encode(x) (up to floating point equality)
    assert torch.allclose(out.z, enc_out.z)
    # x_hat from forward should match decode(z)
    assert torch.allclose(out.x_hat, dec_out.x_hat)


# ================= inference-time behavior =================

def test_ae_inference_encode_decode_use_inference_mode_no_grad():
    batch_size = 3
    in_features = 4
    latent_features = 2

    encoder = SimpleEncoder(in_features=in_features, latent_features=latent_features)
    decoder = SimpleDecoder(latent_features=latent_features, out_features=in_features)
    ae = AE(encoder=encoder, decoder=decoder)

    x = torch.randn(batch_size, in_features)
    ae.build(x)

    # Global grad enabled; encode/decode should still be no-grad due to @torch.inference_mode
    torch.set_grad_enabled(True)
    assert torch.is_grad_enabled() is True

    encode_out = ae.encode(x)
    assert isinstance(encode_out, AEEncodeOutput)
    assert encode_out.z.shape == (batch_size, latent_features)
    assert encode_out.z.requires_grad is False

    decode_out = ae.decode(encode_out.z)
    assert isinstance(decode_out, AEDecodeOutput)
    assert decode_out.x_hat.shape == (batch_size, in_features)
    assert decode_out.x_hat.requires_grad is False

    # encoder/decoder should have seen grad disabled during inference calls
    assert encoder.last_grad_enabled is False
    assert decoder.last_grad_enabled is False

    # Global grad state must be restored after inference_mode
    assert torch.is_grad_enabled() is True


def test_ae_output_repr_uses_modeloutput_smart_repr():
    batch_size = 2
    in_features = 3
    latent_features = 4

    x_hat = torch.randn(batch_size, in_features)
    z = torch.randn(batch_size, latent_features)

    out = AEOutput(x_hat=x_hat, z=z)
    s = repr(out)

    # Check basic structure and tensor summaries
    assert s.startswith("AEOutput(") and s.endswith(")")
    assert "x_hat=Tensor(" in s
    assert f"shape={tuple(x_hat.shape)}" in s
    assert f"dtype={x_hat.dtype}" in s

    assert "z=Tensor(" in s
    assert f"shape={tuple(z.shape)}" in s
    assert f"dtype={z.dtype}" in s
