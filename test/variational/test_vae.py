import pytest
import torch
import torch.nn as nn

from pyautoencoder.variational.vae import (
    VAE,
    VAEEncodeOutput,
    VAEDecodeOutput,
    VAEOutput,
    AdaGVAEOutput,
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

# ================= Build mechanism =================

def test_vae_build_materializes_lazy_layers():
    B, in_features, feat_dim, latent_dim = 3, 5, 7, 4
    x = torch.randn(B, in_features)

    encoder = DummyEncoder(in_features=in_features, feat_dim=feat_dim)
    decoder = DummyDecoder(latent_dim=latent_dim, out_features=in_features)
    vae = VAE(encoder=encoder, decoder=decoder, latent_dim=latent_dim)

    # Before build: sampling_layer mu/log_var are still LazyLinear
    assert isinstance(vae.sampling_layer, FullyFactorizedGaussian)
    assert isinstance(vae.sampling_layer.mu, nn.LazyLinear)
    assert isinstance(vae.sampling_layer.log_var, nn.LazyLinear)

    vae.build(x)

    # After build: LazyLinear has materialized into regular Linear
    assert not isinstance(vae.sampling_layer.mu, nn.LazyLinear)
    assert not isinstance(vae.sampling_layer.log_var, nn.LazyLinear)
    assert vae.sampling_layer.mu.in_features == feat_dim
    assert vae.sampling_layer.log_var.in_features == feat_dim

    # Encoder was called during build
    assert encoder.forward_calls == 1
    assert encoder.last_input_shape == (B, in_features)


def test_vae_build_runs_encoder_under_no_grad():
    B, in_features, feat_dim, latent_dim = 3, 5, 7, 4
    x = torch.randn(B, in_features)

    encoder = DummyEncoder(in_features=in_features, feat_dim=feat_dim)
    decoder = DummyDecoder(latent_dim=latent_dim, out_features=in_features)
    vae = VAE(encoder=encoder, decoder=decoder, latent_dim=latent_dim)

    torch.set_grad_enabled(True)
    vae.build(x)

    assert encoder.last_grad_enabled is False
    assert torch.is_grad_enabled() is True


def test_vae_encode_passes_kwargs_correctly():
    B, in_features, latent_dim, S = 2, 4, 2, 5
    x = torch.randn(B, in_features)
    encoder = DummyEncoder(in_features=in_features, feat_dim=6)
    decoder = DummyDecoder(latent_dim=latent_dim, out_features=in_features)
    vae = VAE(encoder=encoder, decoder=decoder, latent_dim=latent_dim)

    vae.build(x)
    enc = vae._encode(x, S=S)
    assert enc.z.shape == (B, S, latent_dim)


# ================= VAE =================

def test_vae_training_encode_decode_forward_shapes_and_types():
    B, in_features, feat_dim, latent_dim, out_features, S = 4, 6, 8, 3, 5, 7
    x = torch.randn(B, in_features)

    encoder = DummyEncoder(in_features=in_features, feat_dim=feat_dim)
    decoder = DummyDecoder(latent_dim=latent_dim, out_features=out_features)
    vae = VAE(encoder=encoder, decoder=decoder, latent_dim=latent_dim)

    vae.build(x)

    vae.train()
    torch.set_grad_enabled(True)

    enc_out = vae._encode(x, S=S)
    assert isinstance(enc_out, VAEEncodeOutput)
    assert enc_out.z.shape == (B, S, latent_dim)
    assert enc_out.mu.shape == (B, latent_dim)
    assert enc_out.log_var.shape == (B, latent_dim)
    assert enc_out.z.requires_grad is True
    assert enc_out.mu.requires_grad is True
    assert enc_out.log_var.requires_grad is True

    dec_out = vae._decode(enc_out.z)
    assert isinstance(dec_out, VAEDecodeOutput)
    assert dec_out.x_hat.shape == (B, S, out_features)
    assert dec_out.x_hat.requires_grad is True

    out = vae.forward(x, S=S)
    assert isinstance(out, VAEOutput)
    assert out.z.shape == (B, S, latent_dim)
    assert out.mu.shape == (B, latent_dim)
    assert out.log_var.shape == (B, latent_dim)
    assert out.x_hat.shape == (B, S, out_features)
    assert out.x_hat.requires_grad is True
    assert out.z.requires_grad is True

    assert encoder.last_grad_enabled is True
    assert decoder.last_grad_enabled is True

def test_vae_forward_matches_manual_encode_then_decode_in_eval_mode():
    B, in_features, feat_dim, latent_dim, out_features, S = 3, 4, 6, 2, 5, 4
    x = torch.randn(B, in_features)

    encoder = DummyEncoder(in_features=in_features, feat_dim=feat_dim)
    decoder = DummyDecoder(latent_dim=latent_dim, out_features=out_features)
    vae = VAE(encoder=encoder, decoder=decoder, latent_dim=latent_dim)
    vae.build(x)

    vae.eval()
    torch.set_grad_enabled(True)

    enc_out = vae._encode(x, S=S)
    dec_out = vae._decode(enc_out.z)

    out = vae.forward(x, S=S)

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

    z = torch.randn(B, S, latent_dim)

    dec_out = vae._decode(z)

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

    enc1 = vae.encode(x, S=S)
    enc2 = vae.encode(x, S=S)

    assert isinstance(enc1, VAEEncodeOutput)
    assert enc1.z.shape == (B, S, latent_dim)
    assert enc1.mu.shape == (B, latent_dim)
    assert enc1.log_var.shape == (B, latent_dim)

    assert enc1.z.requires_grad is False
    assert enc1.mu.requires_grad is False
    assert enc1.log_var.requires_grad is False

    assert torch.allclose(enc1.z, enc2.z) # type: ignore

    expected_z = enc1.mu.unsqueeze(1).expand(-1, S, -1)
    assert torch.allclose(enc1.z, expected_z)

    assert torch.is_grad_enabled() is True

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

    torch.manual_seed(123)
    enc1 = vae.encode(x, S=S)

    torch.manual_seed(123)
    enc2 = vae.encode(x, S=S)

    assert isinstance(enc1, VAEEncodeOutput)
    assert enc1.z.shape == (B, S, latent_dim)

    assert torch.allclose(enc1.z, enc2.z) # type: ignore

    assert enc1.z.requires_grad is False
    assert enc1.mu.requires_grad is False
    assert enc1.log_var.requires_grad is False

    tiled_mu = enc1.mu.unsqueeze(1).expand(-1, S, -1)
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


# ================= compute_loss =================

def test_vae_compute_loss_gaussian_likelihood_returns_correct_type():
    B, in_features, feat_dim, latent_dim, S = 4, 6, 8, 3, 2
    x = torch.randn(B, in_features)

    encoder = DummyEncoder(in_features=in_features, feat_dim=feat_dim)
    decoder = DummyDecoder(latent_dim=latent_dim, out_features=in_features)
    vae = VAE(encoder=encoder, decoder=decoder, latent_dim=latent_dim)
    vae.build(x)

    vae.train()
    torch.set_grad_enabled(True)

    vae_output = vae.forward(x, S=S)

    from pyautoencoder.loss.base import LossResult
    loss_result = vae.compute_loss(x, vae_output)

    assert isinstance(loss_result, LossResult)
    assert hasattr(loss_result, 'objective')
    assert hasattr(loss_result, 'diagnostics')

    assert loss_result.objective.dim() == 0
    assert loss_result.objective.requires_grad is True

    assert isinstance(loss_result.diagnostics, dict)
    assert 'elbo' in loss_result.diagnostics
    assert 'log_likelihood' in loss_result.diagnostics
    assert 'kl_divergence' in loss_result.diagnostics

    assert isinstance(loss_result.diagnostics['elbo'], float)
    assert isinstance(loss_result.diagnostics['log_likelihood'], float)
    assert isinstance(loss_result.diagnostics['kl_divergence'], float)


def test_vae_compute_loss_objective_is_negative_elbo():
    B, in_features, feat_dim, latent_dim, S = 3, 5, 7, 2, 2
    x = torch.randn(B, in_features)

    encoder = DummyEncoder(in_features=in_features, feat_dim=feat_dim)
    decoder = DummyDecoder(latent_dim=latent_dim, out_features=in_features)
    vae = VAE(encoder=encoder, decoder=decoder, latent_dim=latent_dim)
    vae.build(x)

    vae.train()
    torch.set_grad_enabled(True)

    vae_output = vae.forward(x, S=S)
    loss_result = vae.compute_loss(x, vae_output)

    elbo = loss_result.diagnostics['elbo']
    objective = loss_result.objective.item()
    assert torch.allclose(torch.tensor(objective), torch.tensor(-elbo), atol=1e-6)


def test_vae_compute_loss_with_beta_parameter():
    B, in_features, feat_dim, latent_dim, S = 3, 5, 7, 2, 2
    x = torch.randn(B, in_features)

    encoder = DummyEncoder(in_features=in_features, feat_dim=feat_dim)
    decoder = DummyDecoder(latent_dim=latent_dim, out_features=in_features)
    vae = VAE(encoder=encoder, decoder=decoder, latent_dim=latent_dim)
    vae.build(x)

    vae.train()
    torch.set_grad_enabled(True)

    vae_output = vae.forward(x, S=S)

    loss_beta1 = vae.compute_loss(x, vae_output, beta=1.0)
    loss_beta05 = vae.compute_loss(x, vae_output, beta=0.5)

    assert loss_beta05.diagnostics['elbo'] > loss_beta1.diagnostics['elbo']


def test_vae_compute_loss_kl_divergence_nonnegative():
    B, in_features, feat_dim, latent_dim, S = 3, 5, 7, 2, 2
    x = torch.randn(B, in_features)

    encoder = DummyEncoder(in_features=in_features, feat_dim=feat_dim)
    decoder = DummyDecoder(latent_dim=latent_dim, out_features=in_features)
    vae = VAE(encoder=encoder, decoder=decoder, latent_dim=latent_dim)
    vae.build(x)

    vae.train()
    torch.set_grad_enabled(True)

    vae_output = vae.forward(x, S=S)
    loss_result = vae.compute_loss(x, vae_output)

    assert loss_result.diagnostics['kl_divergence'] >= 0


def test_vae_compute_loss_bernoulli_likelihood():
    B, in_features, feat_dim, latent_dim, S = 3, 5, 7, 2, 2
    x = torch.sigmoid(torch.randn(B, in_features))

    encoder = DummyEncoder(in_features=in_features, feat_dim=feat_dim)
    decoder = DummyDecoder(latent_dim=latent_dim, out_features=in_features)
    vae = VAE(encoder=encoder, decoder=decoder, latent_dim=latent_dim)
    vae.build(x)

    vae.train()
    torch.set_grad_enabled(True)

    vae_output = vae.forward(x, S=S)
    loss_result = vae.compute_loss(x, vae_output, likelihood='bernoulli')

    assert isinstance(loss_result.objective, torch.Tensor)
    assert loss_result.objective.dim() == 0
    assert 'elbo' in loss_result.diagnostics
    assert 'log_likelihood' in loss_result.diagnostics
    assert 'kl_divergence' in loss_result.diagnostics


def test_vae_compute_loss_eval_mode_elbo_independent_of_S():
    B, in_features, feat_dim, latent_dim = 2, 4, 6, 2
    x = torch.randn(B, in_features)

    encoder = DummyEncoder(in_features=in_features, feat_dim=feat_dim)
    decoder = DummyDecoder(latent_dim=latent_dim, out_features=in_features)
    vae = VAE(encoder=encoder, decoder=decoder, latent_dim=latent_dim)
    vae.build(x)

    vae.eval()
    torch.set_grad_enabled(False)

    out_s1 = vae.forward(x, S=1)
    loss_s1 = vae.compute_loss(x, out_s1)

    out_s5 = vae.forward(x, S=5)
    loss_s5 = vae.compute_loss(x, out_s5)

    assert torch.allclose(loss_s1.objective, loss_s5.objective, atol=1e-5)
    assert abs(loss_s1.diagnostics['elbo'] - loss_s5.diagnostics['elbo']) < 1e-5
    assert abs(loss_s1.diagnostics['log_likelihood'] - loss_s5.diagnostics['log_likelihood']) < 1e-5


def test_vae_compute_loss_backward_flows_through_all_params():
    B, in_features, feat_dim, latent_dim, S = 2, 4, 6, 2, 2
    x = torch.randn(B, in_features)

    encoder = DummyEncoder(in_features=in_features, feat_dim=feat_dim)
    decoder = DummyDecoder(latent_dim=latent_dim, out_features=in_features)
    vae = VAE(encoder=encoder, decoder=decoder, latent_dim=latent_dim)
    vae.build(x)

    vae.train()
    torch.set_grad_enabled(True)

    vae_output = vae.forward(x, S=S)
    loss_result = vae.compute_loss(x, vae_output)
    loss_result.objective.backward()

    enc_grads = [p.grad for p in encoder.parameters() if p.requires_grad]
    dec_grads = [p.grad for p in decoder.parameters() if p.requires_grad]
    sl_grads = [p.grad for p in vae.sampling_layer.parameters() if p.requires_grad]

    assert any(g is not None and torch.any(g != 0) for g in enc_grads)
    assert any(g is not None and torch.any(g != 0) for g in dec_grads)
    assert any(g is not None and torch.any(g != 0) for g in sl_grads)


def test_vae_compute_loss_batch_size_one():
    in_features, feat_dim, latent_dim, S = 4, 6, 2, 2
    x = torch.randn(1, in_features)

    encoder = DummyEncoder(in_features=in_features, feat_dim=feat_dim)
    decoder = DummyDecoder(latent_dim=latent_dim, out_features=in_features)
    vae = VAE(encoder=encoder, decoder=decoder, latent_dim=latent_dim)
    vae.build(x)

    vae.train()
    torch.set_grad_enabled(True)

    vae_output = vae.forward(x, S=S)
    loss_result = vae.compute_loss(x, vae_output)

    assert loss_result.objective.dim() == 0
    assert not torch.isnan(loss_result.objective)
    assert not torch.isinf(loss_result.objective)


def test_vae_compute_loss_diagnostics_elbo_consistency():
    B, in_features, feat_dim, latent_dim, S = 3, 5, 7, 2, 2
    x = torch.randn(B, in_features)

    encoder = DummyEncoder(in_features=in_features, feat_dim=feat_dim)
    decoder = DummyDecoder(latent_dim=latent_dim, out_features=in_features)
    vae = VAE(encoder=encoder, decoder=decoder, latent_dim=latent_dim)
    vae.build(x)

    vae.train()
    torch.set_grad_enabled(True)

    vae_output = vae.forward(x, S=S)
    loss_result = vae.compute_loss(x, vae_output)

    ll = loss_result.diagnostics['log_likelihood']
    kl = loss_result.diagnostics['kl_divergence']
    elbo = loss_result.diagnostics['elbo']

    assert abs(elbo - (ll - kl)) < 1e-5


def test_vae_compute_loss_with_different_likelihood_formats():
    B, in_features, feat_dim, latent_dim, S = 3, 5, 7, 2, 2
    x = torch.randn(B, in_features)

    encoder = DummyEncoder(in_features=in_features, feat_dim=feat_dim)
    decoder = DummyDecoder(latent_dim=latent_dim, out_features=in_features)
    vae = VAE(encoder=encoder, decoder=decoder, latent_dim=latent_dim)
    vae.build(x)

    vae.train()
    torch.set_grad_enabled(True)

    vae_output = vae.forward(x, S=S)

    loss_str = vae.compute_loss(x, vae_output, likelihood='gaussian')
    assert isinstance(loss_str.objective, torch.Tensor)

    from pyautoencoder.loss.base import LikelihoodType
    loss_enum = vae.compute_loss(x, vae_output, likelihood=LikelihoodType.GAUSSIAN)
    assert isinstance(loss_enum.objective, torch.Tensor)

    assert torch.allclose(loss_str.objective, loss_enum.objective, atol=1e-6)


# ================= AdaGVAE =================

def test_adagvae_wraps_vae():
    from pyautoencoder.variational.vae import AdaGVAE

    in_features, latent_dim = 6, 3
    encoder = DummyEncoder(in_features=in_features, feat_dim=10)
    decoder = DummyDecoder(latent_dim=latent_dim, out_features=in_features)
    vae = VAE(encoder=encoder, decoder=decoder, latent_dim=latent_dim)

    adagvae = AdaGVAE(vae=vae)

    assert isinstance(adagvae, nn.Module)
    assert not isinstance(adagvae, VAE)
    assert isinstance(adagvae.vae, VAE)


def test_adagvae_forward_training_returns_adagvae_output():
    from pyautoencoder.variational.vae import AdaGVAE

    B, in_features, feat_dim, latent_dim, out_features, S = 4, 6, 8, 3, 5, 2
    x1 = torch.randn(B, in_features)
    x2 = torch.randn(B, in_features)

    encoder = DummyEncoder(in_features=in_features, feat_dim=feat_dim)
    decoder = DummyDecoder(latent_dim=latent_dim, out_features=out_features)
    adagvae = AdaGVAE(vae=VAE(encoder=encoder, decoder=decoder, latent_dim=latent_dim))
    adagvae.build(x1)

    adagvae.train()
    torch.set_grad_enabled(True)

    out = adagvae.forward((x1, x2), S=S)

    assert isinstance(out, AdaGVAEOutput)
    assert isinstance(out.output1, VAEOutput)
    assert isinstance(out.output2, VAEOutput)

    assert out.output1.z.shape == (B, S, latent_dim)
    assert out.output1.mu.shape == (B, latent_dim)
    assert out.output1.log_var.shape == (B, latent_dim)
    assert out.output1.x_hat.shape == (B, S, out_features)

    assert out.output2.z.shape == (B, S, latent_dim)
    assert out.output2.mu.shape == (B, latent_dim)
    assert out.output2.log_var.shape == (B, latent_dim)
    assert out.output2.x_hat.shape == (B, S, out_features)


def test_adagvae_inference_via_encode_decode():
    from pyautoencoder.variational.vae import AdaGVAE

    B, in_features, feat_dim, latent_dim, out_features, S = 4, 6, 8, 3, 5, 2
    x = torch.randn(B, in_features)

    encoder = DummyEncoder(in_features=in_features, feat_dim=feat_dim)
    decoder = DummyDecoder(latent_dim=latent_dim, out_features=out_features)
    adagvae = AdaGVAE(vae=VAE(encoder=encoder, decoder=decoder, latent_dim=latent_dim))
    adagvae.build(x)

    adagvae.eval()

    enc = adagvae.vae.encode(x, S=S)
    dec = adagvae.vae.decode(enc.z)

    assert isinstance(enc, VAEEncodeOutput)
    assert enc.z.shape == (B, S, latent_dim)
    assert enc.z.requires_grad is False

    assert isinstance(dec, VAEDecodeOutput)
    assert dec.x_hat.shape == (B, S, out_features)
    assert dec.x_hat.requires_grad is False


def test_adagvae_encode_pair_returns_correct_shapes():
    from pyautoencoder.variational.vae import AdaGVAE

    B, in_features, feat_dim, latent_dim, S = 3, 5, 7, 2, 3
    x1 = torch.randn(B, in_features)
    x2 = torch.randn(B, in_features)

    encoder = DummyEncoder(in_features=in_features, feat_dim=feat_dim)
    decoder = DummyDecoder(latent_dim=latent_dim, out_features=in_features)
    adagvae = AdaGVAE(vae=VAE(encoder=encoder, decoder=decoder, latent_dim=latent_dim))
    adagvae.build(x1)

    adagvae.train()
    torch.set_grad_enabled(True)

    enc1, enc2 = adagvae._encode_pair(x1, x2, S=S)

    assert isinstance(enc1, VAEEncodeOutput)
    assert isinstance(enc2, VAEEncodeOutput)

    assert enc1.z.shape == (B, S, latent_dim)
    assert enc1.mu.shape == (B, latent_dim)
    assert enc1.log_var.shape == (B, latent_dim)

    assert enc2.z.shape == (B, S, latent_dim)
    assert enc2.mu.shape == (B, latent_dim)
    assert enc2.log_var.shape == (B, latent_dim)


def test_adagvae_compute_loss_returns_correct_structure():
    from pyautoencoder.variational.vae import AdaGVAE
    from pyautoencoder.loss.base import LossResult

    B, in_features, feat_dim, latent_dim, S = 4, 6, 8, 3, 2
    x1 = torch.randn(B, in_features)
    x2 = torch.randn(B, in_features)

    encoder = DummyEncoder(in_features=in_features, feat_dim=feat_dim)
    decoder = DummyDecoder(latent_dim=latent_dim, out_features=in_features)
    adagvae = AdaGVAE(vae=VAE(encoder=encoder, decoder=decoder, latent_dim=latent_dim))
    adagvae.build(x1)

    adagvae.train()
    torch.set_grad_enabled(True)

    out = adagvae.forward((x1, x2), S=S)
    loss_result = adagvae.compute_loss((x1, x2), out)

    assert isinstance(loss_result, LossResult)
    assert loss_result.objective.dim() == 0
    assert loss_result.objective.requires_grad is True

    assert isinstance(loss_result.diagnostics, dict)
    assert 'elbo' in loss_result.diagnostics
    assert 'log_likelihood_x1' in loss_result.diagnostics
    assert 'log_likelihood_x2' in loss_result.diagnostics
    assert 'kl_divergence_x1' in loss_result.diagnostics
    assert 'kl_divergence_x2' in loss_result.diagnostics

    assert isinstance(loss_result.diagnostics['elbo'], float)
    assert isinstance(loss_result.diagnostics['log_likelihood_x1'], float)
    assert isinstance(loss_result.diagnostics['log_likelihood_x2'], float)
    assert isinstance(loss_result.diagnostics['kl_divergence_x1'], float)
    assert isinstance(loss_result.diagnostics['kl_divergence_x2'], float)


def test_adagvae_compute_loss_backward_flows():
    from pyautoencoder.variational.vae import AdaGVAE

    B, in_features, feat_dim, latent_dim, S = 2, 4, 6, 2, 2
    x1 = torch.randn(B, in_features)
    x2 = torch.randn(B, in_features)

    encoder = DummyEncoder(in_features=in_features, feat_dim=feat_dim)
    decoder = DummyDecoder(latent_dim=latent_dim, out_features=in_features)
    adagvae = AdaGVAE(vae=VAE(encoder=encoder, decoder=decoder, latent_dim=latent_dim))
    adagvae.build(x1)

    adagvae.train()
    torch.set_grad_enabled(True)

    out = adagvae.forward((x1, x2), S=S)
    loss_result = adagvae.compute_loss((x1, x2), out)
    loss_result.objective.backward()

    enc_grads = [p.grad for p in encoder.parameters() if p.requires_grad]
    dec_grads = [p.grad for p in decoder.parameters() if p.requires_grad]
    sl_grads = [p.grad for p in adagvae.vae.sampling_layer.parameters() if p.requires_grad]

    assert any(g is not None and torch.any(g != 0) for g in enc_grads)
    assert any(g is not None and torch.any(g != 0) for g in dec_grads)
    assert any(g is not None and torch.any(g != 0) for g in sl_grads)


def test_adagvae_compute_loss_with_beta():
    from pyautoencoder.variational.vae import AdaGVAE

    B, in_features, feat_dim, latent_dim, S = 3, 5, 7, 2, 2
    x1 = torch.randn(B, in_features)
    x2 = torch.randn(B, in_features)

    encoder = DummyEncoder(in_features=in_features, feat_dim=feat_dim)
    decoder = DummyDecoder(latent_dim=latent_dim, out_features=in_features)
    adagvae = AdaGVAE(vae=VAE(encoder=encoder, decoder=decoder, latent_dim=latent_dim))
    adagvae.build(x1)

    adagvae.train()
    torch.set_grad_enabled(True)

    out = adagvae.forward((x1, x2), S=S)

    loss_beta1 = adagvae.compute_loss((x1, x2), out, beta=1.0)
    loss_beta05 = adagvae.compute_loss((x1, x2), out, beta=0.5)

    assert loss_beta05.diagnostics['elbo'] > loss_beta1.diagnostics['elbo']


def test_adagvae_identical_inputs_produce_no_grouping():
    """When x1 == x2, KL(q1||q2) = 0 everywhere, tau = 0, mask is all-False.
    The adapted posteriors must equal the individual (unadapted) posteriors."""
    from pyautoencoder.variational.vae import AdaGVAE

    B, in_features, feat_dim, latent_dim = 3, 5, 7, 4
    x = torch.randn(B, in_features)

    encoder = DummyEncoder(in_features=in_features, feat_dim=feat_dim)
    decoder = DummyDecoder(latent_dim=latent_dim, out_features=in_features)
    adagvae = AdaGVAE(vae=VAE(encoder=encoder, decoder=decoder, latent_dim=latent_dim))
    adagvae.build(x)

    adagvae.eval()
    torch.set_grad_enabled(False)

    enc1_pair, enc2_pair = adagvae._encode_pair(x, x.clone(), S=1)
    single_enc = adagvae.vae._encode(x, S=1)

    assert torch.allclose(enc1_pair.mu, single_enc.mu, atol=1e-6)
    assert torch.allclose(enc1_pair.log_var, single_enc.log_var, atol=1e-6)
    assert torch.allclose(enc2_pair.mu, single_enc.mu, atol=1e-6)
    assert torch.allclose(enc2_pair.log_var, single_enc.log_var, atol=1e-6)


def test_adagvae_encode_pair_with_different_s():
    from pyautoencoder.variational.vae import AdaGVAE

    B, in_features, feat_dim, latent_dim = 3, 5, 7, 2
    x1 = torch.randn(B, in_features)
    x2 = torch.randn(B, in_features)

    encoder = DummyEncoder(in_features=in_features, feat_dim=feat_dim)
    decoder = DummyDecoder(latent_dim=latent_dim, out_features=in_features)
    adagvae = AdaGVAE(vae=VAE(encoder=encoder, decoder=decoder, latent_dim=latent_dim))
    adagvae.build(x1)

    adagvae.eval()
    torch.set_grad_enabled(False)

    enc1_s1, enc2_s1 = adagvae._encode_pair(x1, x2, S=1)
    assert enc1_s1.z.shape == (B, 1, latent_dim)
    assert enc2_s1.z.shape == (B, 1, latent_dim)

    enc1_s5, enc2_s5 = adagvae._encode_pair(x1, x2, S=5)
    assert enc1_s5.z.shape == (B, 5, latent_dim)
    assert enc2_s5.z.shape == (B, 5, latent_dim)


def test_adagvae_compute_loss_bernoulli_likelihood():
    from pyautoencoder.variational.vae import AdaGVAE

    B, in_features, feat_dim, latent_dim, S = 3, 5, 7, 2, 2
    x1 = torch.sigmoid(torch.randn(B, in_features))
    x2 = torch.sigmoid(torch.randn(B, in_features))

    encoder = DummyEncoder(in_features=in_features, feat_dim=feat_dim)
    decoder = DummyDecoder(latent_dim=latent_dim, out_features=in_features)
    adagvae = AdaGVAE(vae=VAE(encoder=encoder, decoder=decoder, latent_dim=latent_dim))
    adagvae.build(x1)

    adagvae.train()
    torch.set_grad_enabled(True)

    out = adagvae.forward((x1, x2), S=S)
    loss_result = adagvae.compute_loss((x1, x2), out, likelihood='bernoulli')

    assert isinstance(loss_result.objective, torch.Tensor)
    assert loss_result.objective.dim() == 0
    assert 'elbo' in loss_result.diagnostics


def test_adagvae_compute_loss_equals_sum_of_individual_losses():
    from pyautoencoder.variational.vae import AdaGVAE

    B, in_features, feat_dim, latent_dim, S = 3, 5, 7, 2, 2
    x1 = torch.randn(B, in_features)
    x2 = torch.randn(B, in_features)

    encoder = DummyEncoder(in_features=in_features, feat_dim=feat_dim)
    decoder = DummyDecoder(latent_dim=latent_dim, out_features=in_features)
    adagvae = AdaGVAE(vae=VAE(encoder=encoder, decoder=decoder, latent_dim=latent_dim))
    adagvae.build(x1)

    adagvae.train()
    torch.set_grad_enabled(True)

    out = adagvae.forward((x1, x2), S=S)
    pair_loss = adagvae.compute_loss((x1, x2), out)

    loss1 = adagvae.vae.compute_loss(x=x1, vae_output=out.output1)
    loss2 = adagvae.vae.compute_loss(x=x2, vae_output=out.output2)

    assert torch.allclose(pair_loss.objective, loss1.objective + loss2.objective, atol=1e-5)
    assert abs(
        pair_loss.diagnostics['elbo']
        - (loss1.diagnostics['elbo'] + loss2.diagnostics['elbo'])
    ) < 1e-5
