import pytest
import torch
import torch.nn as nn

from pyautoencoder._base.base import NotBuiltError
from pyautoencoder.variational.vae import (
    VAE,
    VAEEncodeOutput,
    VAEDecodeOutput,
    VAEOutput,
)
from pyautoencoder.variational.stochastic_layers import FullyFactorizedGaussian

class DummyEncoder(nn.Module):
    def __init__(self, in_features: int, feat_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_features, feat_dim, bias=False)
        self.last_input_shape = None
        self.forward_calls = 0
        self.last_grad_enabled: bool | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        self.last_input_shape = tuple(x.shape)
        self.last_grad_enabled = torch.is_grad_enabled()
        return self.linear(x)


class DummyDecoder(nn.Module):
    def __init__(self, latent_dim: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(latent_dim, out_features, bias=False)
        self.last_input_shape = None
        self.forward_calls = 0
        self.last_grad_enabled: bool | None = None

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        self.last_input_shape = tuple(z.shape)
        self.last_grad_enabled = torch.is_grad_enabled()
        return self.linear(z)

# ================= VAE =================

def test_vae_raises_before_build_on_all_guarded_paths():
    B, in_features, latent_dim = 4, 6, 3
    encoder = DummyEncoder(in_features=in_features, feat_dim=10)
    decoder = DummyDecoder(latent_dim=latent_dim, out_features=in_features)
    vae = VAE(encoder=encoder, decoder=decoder, latent_dim=latent_dim)

    x = torch.randn(B, in_features)
    z = torch.randn(B, 1, latent_dim)

    with pytest.raises(NotBuiltError):
        vae.forward(x)

    with pytest.raises(NotBuiltError):
        vae._encode(x)

    with pytest.raises(NotBuiltError):
        vae._decode(z)

    with pytest.raises(NotBuiltError):
        vae.encode(x)

    with pytest.raises(NotBuiltError):
        vae.decode(z)

def test_vae_build_initializes_sampling_layer_and_sets_flag_idempotent():
    B, in_features, feat_dim, latent_dim = 3, 5, 7, 4
    x = torch.randn(B, in_features)

    encoder = DummyEncoder(in_features=in_features, feat_dim=feat_dim)
    decoder = DummyDecoder(latent_dim=latent_dim, out_features=in_features)
    vae = VAE(encoder=encoder, decoder=decoder, latent_dim=latent_dim)

    # sampling_layer is present but not built
    assert isinstance(vae.sampling_layer, FullyFactorizedGaussian)
    assert vae.sampling_layer.built is False
    assert vae.built is False

    # Build once
    vae.build(x)

    # VAE and sampling layer must be built
    assert vae.built is True
    assert vae._built is True
    assert vae.sampling_layer.built is True

    # Encoder must have been called once during build
    assert encoder.forward_calls == 1
    assert encoder.last_input_shape == (B, in_features)

    # sampling_layer should have in_features equal to encoder output size
    assert vae.sampling_layer.in_features == feat_dim

    # Build should not replace encoder/decoder
    enc_id = id(vae.encoder)
    dec_id = id(vae.decoder)

    # Second build: BuildGuardMixin wrapper should early-return, no extra calls
    vae.build(x)
    assert id(vae.encoder) == enc_id
    assert id(vae.decoder) == dec_id
    assert encoder.forward_calls == 1  # no additional encode call on second build

def test_vae_training_encode_decode_forward_shapes_and_types():
    B, in_features, feat_dim, latent_dim, out_features, S = 4, 6, 8, 3, 5, 7
    x = torch.randn(B, in_features)

    encoder = DummyEncoder(in_features=in_features, feat_dim=feat_dim)
    decoder = DummyDecoder(latent_dim=latent_dim, out_features=out_features)
    vae = VAE(encoder=encoder, decoder=decoder, latent_dim=latent_dim)

    vae.build(x)

    vae.train()
    torch.set_grad_enabled(True)

    # _encode
    enc_out = vae._encode(x, S=S)
    assert isinstance(enc_out, VAEEncodeOutput)
    assert enc_out.z.shape == (B, S, latent_dim)
    assert enc_out.mu.shape == (B, latent_dim)
    assert enc_out.log_var.shape == (B, latent_dim)
    assert enc_out.z.requires_grad is True
    assert enc_out.mu.requires_grad is True
    assert enc_out.log_var.requires_grad is True

    # _decode
    dec_out = vae._decode(enc_out.z)
    assert isinstance(dec_out, VAEDecodeOutput)
    # Output shape: [B, S, ...] where ... = out_features
    assert dec_out.x_hat.shape == (B, S, out_features)
    assert dec_out.x_hat.requires_grad is True

    # forward
    out = vae.forward(x, S=S)
    assert isinstance(out, VAEOutput)
    assert out.z.shape == (B, S, latent_dim)
    assert out.mu.shape == (B, latent_dim)
    assert out.log_var.shape == (B, latent_dim)
    assert out.x_hat.shape == (B, S, out_features)
    assert out.x_hat.requires_grad is True
    assert out.z.requires_grad is True

    # encoder/decoder should see grad enabled during training
    assert encoder.last_grad_enabled is True
    assert decoder.last_grad_enabled is True

def test_vae_forward_matches_manual_encode_then_decode_in_eval_mode():
    B, in_features, feat_dim, latent_dim, out_features, S = 3, 4, 6, 2, 5, 4
    x = torch.randn(B, in_features)

    encoder = DummyEncoder(in_features=in_features, feat_dim=feat_dim)
    decoder = DummyDecoder(latent_dim=latent_dim, out_features=out_features)
    vae = VAE(encoder=encoder, decoder=decoder, latent_dim=latent_dim)
    vae.build(x)

    # Eval mode: sampling_layer becomes deterministic (z is tiled mu)
    vae.eval()
    torch.set_grad_enabled(True)

    enc_out = vae._encode(x, S=S)
    dec_out = vae._decode(enc_out.z)

    out = vae.forward(x, S=S)

    # In eval mode, forward should be equivalent to encode -> decode
    assert torch.allclose(out.z, enc_out.z)
    assert torch.allclose(out.mu, enc_out.mu)
    assert torch.allclose(out.log_var, enc_out.log_var)
    assert torch.allclose(out.x_hat, dec_out.x_hat)


def test_vae_backward_updates_encoder_and_decoder_params():
    B, in_features, feat_dim, latent_dim, out_features, S = 2, 3, 5, 2, 4, 3
    x = torch.randn(B, in_features)

    encoder = DummyEncoder(in_features=in_features, feat_dim=feat_dim)
    decoder = DummyDecoder(latent_dim=latent_dim, out_features=out_features)
    vae = VAE(encoder=encoder, decoder=decoder, latent_dim=latent_dim)
    vae.build(x)

    vae.train()
    torch.set_grad_enabled(True)

    out = vae.forward(x, S=S)
    loss = out.x_hat.sum()
    loss.backward()

    # Some encoder params should have non-zero gradients
    enc_grads = [p.grad for p in encoder.parameters() if p.requires_grad]
    dec_grads = [p.grad for p in decoder.parameters() if p.requires_grad]
    sl_grads = [p.grad for p in vae.sampling_layer.parameters() if p.requires_grad]

    assert any(g is not None and torch.any(g != 0) for g in enc_grads)
    assert any(g is not None and torch.any(g != 0) for g in dec_grads)
    assert any(g is not None and torch.any(g != 0) for g in sl_grads)

def test_vae_decode_reshapes_consistently_with_flattened_decoder_call():
    B, in_features, feat_dim, latent_dim, out_features, S = 3, 4, 6, 2, 7, 5
    x = torch.randn(B, in_features)

    encoder = DummyEncoder(in_features=in_features, feat_dim=feat_dim)
    decoder = DummyDecoder(latent_dim=latent_dim, out_features=out_features)
    vae = VAE(encoder=encoder, decoder=decoder, latent_dim=latent_dim)
    vae.build(x)

    vae.train()
    torch.set_grad_enabled(True)

    # Create some arbitrary z
    z = torch.randn(B, S, latent_dim)

    # VAE path
    dec_out = vae._decode(z)

    # Manual decode: flatten [B, S, Dz] -> [B*S, Dz], then reshape back
    z_flat = z.reshape(B * S, latent_dim)
    x_hat_flat_manual = decoder(z_flat)
    x_hat_manual = x_hat_flat_manual.reshape(B, S, out_features)

    assert torch.allclose(dec_out.x_hat, x_hat_manual)
    assert decoder.last_input_shape == (B * S, latent_dim)

def test_vae_encode_decode_in_eval_mode_are_deterministic_and_no_grad():
    B, in_features, feat_dim, latent_dim, out_features, S = 3, 4, 6, 2, 5, 7
    x = torch.randn(B, in_features)

    encoder = DummyEncoder(in_features=in_features, feat_dim=feat_dim)
    decoder = DummyDecoder(latent_dim=latent_dim, out_features=out_features)
    vae = VAE(encoder=encoder, decoder=decoder, latent_dim=latent_dim)
    vae.build(x)

    vae.eval()
    torch.set_grad_enabled(True)

    # encode is @torch.inference_mode() in BaseAutoencoder
    enc1 = vae.encode(x, S=S)
    enc2 = vae.encode(x, S=S)

    assert isinstance(enc1, VAEEncodeOutput)
    assert enc1.z.shape == (B, S, latent_dim)
    assert enc1.mu.shape == (B, latent_dim)
    assert enc1.log_var.shape == (B, latent_dim)

    # No gradients from encode in eval mode
    assert enc1.z.requires_grad is False
    assert enc1.mu.requires_grad is False
    assert enc1.log_var.requires_grad is False

    # Deterministic: sampling_layer.eval() should tile mu, no randomness
    assert torch.allclose(enc1.z, enc2.z) # type: ignore

    # z should be mu expanded along S
    expected_z = enc1.mu.unsqueeze(1).expand(-1, S, -1)
    assert torch.allclose(enc1.z, expected_z)

    # Global grad state should be restored
    assert torch.is_grad_enabled() is True

    # decode is also inference_mode
    dec_out = vae.decode(enc1.z)
    assert isinstance(dec_out, VAEDecodeOutput)
    assert dec_out.x_hat.shape == (B, S, out_features)
    assert dec_out.x_hat.requires_grad is False

def test_vae_encode_in_train_mode_still_samples_but_without_grad():
    B, in_features, feat_dim, latent_dim, out_features, S = 4, 5, 7, 3, 6, 2
    x = torch.randn(B, in_features)

    encoder = DummyEncoder(in_features=in_features, feat_dim=feat_dim)
    decoder = DummyDecoder(latent_dim=latent_dim, out_features=out_features)
    vae = VAE(encoder=encoder, decoder=decoder, latent_dim=latent_dim)
    vae.build(x)

    vae.train()
    torch.set_grad_enabled(True)

    # Because encode() is inference_mode, no grad, but training flag = True => sampling
    torch.manual_seed(123)
    enc1 = vae.encode(x, S=S)

    torch.manual_seed(123)
    enc2 = vae.encode(x, S=S)

    assert isinstance(enc1, VAEEncodeOutput)
    assert enc1.z.shape == (B, S, latent_dim)

    # Still stochastic but reproducible with same seed
    assert torch.allclose(enc1.z, enc2.z) # type: ignore

    # No grad due to inference_mode
    assert enc1.z.requires_grad is False
    assert enc1.mu.requires_grad is False
    assert enc1.log_var.requires_grad is False

    # training mode implies sampling_layer.training == True
    # so z should *not* equal mu tiling in general
    tiled_mu = enc1.mu.unsqueeze(1).expand(-1, S, -1)
    # Not a strict guarantee, but extremely likely to differ
    assert not torch.allclose(enc1.z, tiled_mu)

def test_vae_output_repr_uses_modeloutput_smart_repr():
    B, S, Dz, out_features = 2, 3, 4, 5
    x_hat = torch.randn(B, S, out_features)
    z = torch.randn(B, S, Dz)
    mu = torch.randn(B, Dz)
    log_var = torch.randn(B, Dz)

    out = VAEOutput(x_hat=x_hat, z=z, mu=mu, log_var=log_var)
    s = repr(out)

    assert s.startswith("VAEOutput(") and s.endswith(")")
    assert "x_hat=Tensor(" in s
    assert f"shape={tuple(x_hat.shape)}" in s
    assert "z=Tensor(" in s
    assert f"shape={tuple(z.shape)}" in s
    assert "mu=Tensor(" in s
    assert f"shape={tuple(mu.shape)}" in s
    assert "log_var=Tensor(" in s
    assert f"shape={tuple(log_var.shape)}" in s

