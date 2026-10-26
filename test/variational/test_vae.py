import math

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


def _is_detached_scalar(t) -> bool:
    return isinstance(t, torch.Tensor) and t.dim() == 0 and not t.requires_grad

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

def test_vae_forward_matches_manual_encode_then_decode():
    B, in_features, feat_dim, latent_dim, out_features, S = 3, 4, 6, 2, 5, 4
    x = torch.randn(B, in_features)

    encoder = DummyEncoder(in_features=in_features, feat_dim=feat_dim)
    decoder = DummyDecoder(latent_dim=latent_dim, out_features=out_features)
    vae = VAE(encoder=encoder, decoder=decoder, latent_dim=latent_dim)
    vae.build(x)

    torch.manual_seed(0)
    enc_out = vae._encode(x, S=S)
    dec_out = vae._decode(enc_out.z)

    torch.manual_seed(0)
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

    z = torch.randn(B, S, latent_dim)

    dec_out = vae._decode(z)

    z_flat = z.reshape(B * S, latent_dim)
    x_hat_flat_manual = decoder(z_flat)
    x_hat_manual = x_hat_flat_manual.reshape(B, S, out_features)

    assert torch.allclose(dec_out.x_hat, x_hat_manual)
    assert decoder.last_input_shape == (B * S, latent_dim)

@pytest.mark.parametrize("training", [True, False])
def test_vae_encode_defaults_to_posterior_mean_without_grad(training):
    B, in_features, feat_dim, latent_dim, out_features = 3, 4, 6, 2, 5
    x = torch.randn(B, in_features)

    encoder = DummyEncoder(in_features=in_features, feat_dim=feat_dim)
    decoder = DummyDecoder(latent_dim=latent_dim, out_features=out_features)
    vae = VAE(encoder=encoder, decoder=decoder, latent_dim=latent_dim)
    vae.build(x)
    vae.train(training)

    enc1 = vae.encode(x)
    enc2 = vae.encode(x)

    assert isinstance(enc1, VAEEncodeOutput)
    assert enc1.z.shape == (B, 1, latent_dim)
    assert enc1.mu.shape == (B, latent_dim)
    assert enc1.log_var.shape == (B, latent_dim)
    assert torch.equal(enc1.z, enc1.mu.unsqueeze(1))
    assert torch.equal(enc1.z, enc2.z)

    assert enc1.z.requires_grad is False
    assert enc1.mu.requires_grad is False
    assert enc1.log_var.requires_grad is False
    assert torch.is_grad_enabled() is True

    dec_out = vae.decode(enc1.z)
    assert isinstance(dec_out, VAEDecodeOutput)
    assert dec_out.x_hat.shape == (B, 1, out_features)
    assert dec_out.x_hat.requires_grad is False


def test_vae_encode_with_sample_true_draws_samples_without_grad():
    B, in_features, feat_dim, latent_dim, out_features, S = 4, 5, 7, 3, 6, 2
    x = torch.randn(B, in_features)

    encoder = DummyEncoder(in_features=in_features, feat_dim=feat_dim)
    decoder = DummyDecoder(latent_dim=latent_dim, out_features=out_features)
    vae = VAE(encoder=encoder, decoder=decoder, latent_dim=latent_dim)
    vae.build(x)
    vae.eval()  # sampling does not depend on the mode

    torch.manual_seed(123)
    enc1 = vae.encode(x, S=S, sample=True)

    torch.manual_seed(123)
    enc2 = vae.encode(x, S=S, sample=True)

    assert isinstance(enc1, VAEEncodeOutput)
    assert enc1.z.shape == (B, S, latent_dim)
    assert torch.allclose(enc1.z, enc2.z)

    assert enc1.z.requires_grad is False
    assert enc1.mu.requires_grad is False
    assert enc1.log_var.requires_grad is False

    tiled_mu = enc1.mu.unsqueeze(1).expand(-1, S, -1)
    assert not torch.allclose(enc1.z, tiled_mu)


def test_vae_encode_with_several_samples_requires_sample_true():
    vae = VAE(encoder=DummyEncoder(4, 6), decoder=DummyDecoder(2, 4), latent_dim=2)

    with pytest.raises(ValueError, match="sample=True"):
        vae.encode(torch.randn(3, 4), S=5)


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

    assert _is_detached_scalar(loss_result.diagnostics['elbo'])
    assert _is_detached_scalar(loss_result.diagnostics['log_likelihood'])
    assert _is_detached_scalar(loss_result.diagnostics['kl_divergence'])


def test_vae_compute_loss_objective_is_negative_elbo():
    B, in_features, feat_dim, latent_dim, S = 3, 5, 7, 2, 2
    x = torch.randn(B, in_features)

    encoder = DummyEncoder(in_features=in_features, feat_dim=feat_dim)
    decoder = DummyDecoder(latent_dim=latent_dim, out_features=in_features)
    vae = VAE(encoder=encoder, decoder=decoder, latent_dim=latent_dim)
    vae.build(x)

    vae.train()

    vae_output = vae.forward(x, S=S)
    loss_result = vae.compute_loss(x, vae_output)

    elbo = loss_result.diagnostics['elbo']
    objective = loss_result.objective.item()
    assert torch.allclose(torch.tensor(objective), -elbo, atol=1e-6)


def test_vae_compute_loss_with_beta_parameter():
    B, in_features, feat_dim, latent_dim, S = 3, 5, 7, 2, 2
    x = torch.randn(B, in_features)

    encoder = DummyEncoder(in_features=in_features, feat_dim=feat_dim)
    decoder = DummyDecoder(latent_dim=latent_dim, out_features=in_features)
    vae = VAE(encoder=encoder, decoder=decoder, latent_dim=latent_dim)
    vae.build(x)

    vae.train()

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

    vae_output = vae.forward(x, S=S)
    loss_result = vae.compute_loss(x, vae_output, likelihood='bernoulli')

    assert isinstance(loss_result.objective, torch.Tensor)
    assert loss_result.objective.dim() == 0
    assert 'elbo' in loss_result.diagnostics
    assert 'log_likelihood' in loss_result.diagnostics
    assert 'kl_divergence' in loss_result.diagnostics


def test_vae_forward_samples_in_eval_mode_too():
    B, in_features, feat_dim, latent_dim = 2, 4, 6, 2
    x = torch.randn(B, in_features)

    encoder = DummyEncoder(in_features=in_features, feat_dim=feat_dim)
    decoder = DummyDecoder(latent_dim=latent_dim, out_features=in_features)
    vae = VAE(encoder=encoder, decoder=decoder, latent_dim=latent_dim)
    vae.build(x)

    with torch.no_grad():
        vae.train()
        torch.manual_seed(0)
        loss_train = vae.compute_loss(x, vae(x, S=5))

        # Without dropout/batch norm, eval mode gives exactly the same sampled ELBO
        vae.eval()
        torch.manual_seed(0)
        out_eval = vae(x, S=5)
        loss_eval = vae.compute_loss(x, out_eval)

        out_mean = vae(x, sample=False)

    assert not torch.allclose(out_eval.z, out_eval.mu.unsqueeze(1).expand(-1, 5, -1))
    assert torch.equal(loss_train.diagnostics['elbo'], loss_eval.diagnostics['elbo'])
    assert torch.equal(out_mean.z, out_mean.mu.unsqueeze(1))


def test_vae_compute_loss_backward_flows_through_all_params():
    B, in_features, feat_dim, latent_dim, S = 2, 4, 6, 2, 2
    x = torch.randn(B, in_features)

    encoder = DummyEncoder(in_features=in_features, feat_dim=feat_dim)
    decoder = DummyDecoder(latent_dim=latent_dim, out_features=in_features)
    vae = VAE(encoder=encoder, decoder=decoder, latent_dim=latent_dim)
    vae.build(x)

    vae.train()

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

    vae_output = vae.forward(x, S=S)

    loss_str = vae.compute_loss(x, vae_output, likelihood='gaussian')
    assert isinstance(loss_str.objective, torch.Tensor)

    from pyautoencoder.loss.base import LikelihoodType
    loss_enum = vae.compute_loss(x, vae_output, likelihood=LikelihoodType.GAUSSIAN)
    assert isinstance(loss_enum.objective, torch.Tensor)

    assert torch.allclose(loss_str.objective, loss_enum.objective, atol=1e-6)


# ================= Save / load =================

def test_vae_state_dict_round_trip_without_build():
    B, in_features, feat_dim, latent_dim = 3, 5, 7, 4
    x = torch.randn(B, in_features)

    def make_vae() -> VAE:
        encoder = DummyEncoder(in_features=in_features, feat_dim=feat_dim)
        decoder = DummyDecoder(latent_dim=latent_dim, out_features=in_features)
        return VAE(encoder=encoder, decoder=decoder, latent_dim=latent_dim)

    src = make_vae()
    src.build(x)

    # The sampling layer is still lazy here; loading materializes it
    dst = make_vae()
    dst.load_state_dict(src.state_dict(), strict=True)

    with torch.no_grad():
        out_src = src(x, sample=False)
        out_dst = dst(x, sample=False)
    assert torch.equal(out_src.mu, out_dst.mu)
    assert torch.equal(out_src.log_var, out_dst.log_var)
    assert torch.equal(out_src.x_hat, out_dst.x_hat)


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

    B, in_features, feat_dim, latent_dim, out_features = 4, 6, 8, 3, 5
    x = torch.randn(B, in_features)

    encoder = DummyEncoder(in_features=in_features, feat_dim=feat_dim)
    decoder = DummyDecoder(latent_dim=latent_dim, out_features=out_features)
    adagvae = AdaGVAE(vae=VAE(encoder=encoder, decoder=decoder, latent_dim=latent_dim))
    adagvae.build(x)

    adagvae.eval()

    enc = adagvae.vae.encode(x)
    dec = adagvae.vae.decode(enc.z)

    assert isinstance(enc, VAEEncodeOutput)
    assert enc.z.shape == (B, 1, latent_dim)
    assert torch.equal(enc.z, enc.mu.unsqueeze(1))
    assert enc.z.requires_grad is False

    assert isinstance(dec, VAEDecodeOutput)
    assert dec.x_hat.shape == (B, 1, out_features)
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

    assert _is_detached_scalar(loss_result.diagnostics['elbo'])
    assert _is_detached_scalar(loss_result.diagnostics['log_likelihood_x1'])
    assert _is_detached_scalar(loss_result.diagnostics['log_likelihood_x2'])
    assert _is_detached_scalar(loss_result.diagnostics['kl_divergence_x1'])
    assert _is_detached_scalar(loss_result.diagnostics['kl_divergence_x2'])


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
    with torch.no_grad():
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
    with torch.no_grad():
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

    out = adagvae.forward((x1, x2), S=S)
    pair_loss = adagvae.compute_loss((x1, x2), out)

    # Expected value from the closed-form formulas, independently of VAE.compute_loss:
    # Gaussian log-likelihood (unit variance) averaged over S, KL to N(0, I).
    def elbo(x, o):
        log_px_z = -0.5 * ((o.x_hat - x.unsqueeze(1)) ** 2).sum(-1)              # [B, S]
        kl = 0.5 * (o.log_var.exp() + o.mu ** 2 - 1 - o.log_var).sum(-1)         # [B]
        return (log_px_z.mean(1) - kl).mean()

    expected_elbo = elbo(x1, out.output1) + elbo(x2, out.output2)

    assert torch.allclose(pair_loss.objective, -expected_elbo, atol=1e-5)
    assert torch.allclose(pair_loss.diagnostics['elbo'], expected_elbo.detach(), atol=1e-5)


def test_adagvae_encode_pair_groups_dimensions_below_kl_threshold(monkeypatch):
    """Adaptive grouping against hand-computed values (Locatello et al., 2020)."""
    from pyautoencoder.variational.vae import AdaGVAE

    D = 4
    adagvae = AdaGVAE(vae=_make_built_vae(latent_dim=D))
    # Bypass encoder and heads: each input row is the concatenation [mu | log_var].
    adagvae.vae.encoder = nn.Identity()
    monkeypatch.setattr(adagvae.vae.sampling_layer, "get_params", lambda f: (f[:, :D], f[:, D:]))

    log4, log2_5 = math.log(4.0), math.log(2.5)
    # Row 0: KL(q1||q2) per dim = [0, 0.5*(3 - log 4), 4.5, 12.5] -> tau = 6.25,
    #        so dims 0-2 are averaged and dim 3 is kept.
    # Row 1: KL = 0.5 on every dim -> no dim is strictly below tau -> no grouping.
    mu1 = torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    lv1 = torch.tensor([[0.0, log4, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    mu2 = torch.tensor([[0.0, 0.0, 3.0, 5.0], [1.0, 1.0, 1.0, 1.0]])
    lv2 = torch.zeros(2, D)

    enc1, enc2 = adagvae._encode_pair(torch.cat([mu1, lv1], 1), torch.cat([mu2, lv2], 1), S=3)

    # Averaged dims: mean of the means, log of the mean of the variances.
    assert torch.allclose(enc1.mu[0], torch.tensor([0.0, 0.0, 1.5, 0.0]))
    assert torch.allclose(enc2.mu[0], torch.tensor([0.0, 0.0, 1.5, 5.0]))
    assert torch.allclose(enc1.log_var[0], torch.tensor([0.0, log2_5, 0.0, 0.0]))
    assert torch.allclose(enc2.log_var[0], torch.tensor([0.0, log2_5, 0.0, 0.0]))

    assert torch.equal(enc1.mu[1], mu1[1]) and torch.equal(enc2.mu[1], mu2[1])
    assert torch.equal(enc1.log_var[1], lv1[1]) and torch.equal(enc2.log_var[1], lv2[1])

    assert enc1.z.shape == enc2.z.shape == (2, 3, D)


# ================= Shape validation =================

def _make_built_vae(in_features: int = 5, feat_dim: int = 7, latent_dim: int = 4) -> VAE:
    encoder = DummyEncoder(in_features=in_features, feat_dim=feat_dim)
    decoder = DummyDecoder(latent_dim=latent_dim, out_features=in_features)
    vae = VAE(encoder=encoder, decoder=decoder, latent_dim=latent_dim)
    vae.build(torch.randn(2, in_features))
    return vae


@pytest.mark.parametrize("z_shape", [(3, 4), (3, 1, 5), (3, 1, 1, 4)])
def test_vae_decode_rejects_z_without_sample_dim_or_wrong_latent_dim(z_shape):
    vae = _make_built_vae(latent_dim=4)

    with pytest.raises(ValueError, match=r"\[B, S, D_z\].*D_z = 4.*unsqueeze\(1\)"):
        vae.decode(torch.randn(z_shape))
    with pytest.raises(ValueError, match=r"\[B, S, D_z\]"):
        vae._decode(torch.randn(z_shape))


@pytest.mark.parametrize(
    "x_shape, dec_out_shape, S",
    [
        ((4, 1, 2, 2), (2, 2), 1),  # decoder drops the channel dim: x_hat has the shape of x
        ((4, 1, 2, 2), (2, 2), 5),  # same decoder, S = 5: samples were summed, not averaged
        ((4, 3, 2, 2), (2, 2), 3),  # RGB with S equal to the number of channels
        ((3, 4), (1, 4), 3),        # extra dim with B == S: samples were compared with other inputs
        ((3, 4), (1, 4), 2),        # extra dim with B != S
    ],
)
def test_vae_compute_loss_rejects_decoder_output_not_matching_x(x_shape, dec_out_shape, S):
    encoder = nn.Sequential(nn.Flatten(), nn.Linear(math.prod(x_shape[1:]), 6))
    decoder = nn.Sequential(
        nn.Linear(2, math.prod(dec_out_shape)),
        nn.Unflatten(-1, dec_out_shape),
    )
    vae = VAE(encoder=encoder, decoder=decoder, latent_dim=2)
    x = torch.rand(x_shape)
    vae.build(x)

    out = vae(x, S=S)

    with pytest.raises(ValueError, match=r"x_hat must have shape \[B, S, \*x\.shape\[1:\]\]"):
        vae.compute_loss(x, out)


@pytest.mark.parametrize(
    "x_hat_shape, suggests_unsqueeze",
    [
        ((3, 5), True),      # no sample dimension
        ((3, 0, 5), False),  # S = 0
        ((4, 2, 5), False),  # batch differs from x
        ((3, 2, 6), False),  # one sample differs from one input
    ],
)
def test_vae_compute_loss_rejects_bad_x_hat_shape(x_hat_shape, suggests_unsqueeze):
    vae = _make_built_vae(in_features=5, latent_dim=4)
    x = torch.randn(3, 5)
    out = VAEOutput(
        x_hat=torch.randn(x_hat_shape),
        z=torch.randn(3, 2, 4),
        mu=torch.randn(3, 4),
        log_var=torch.randn(3, 4),
    )

    with pytest.raises(ValueError, match=r"= \[3, S, 5\] with S >= 1") as exc_info:
        vae.compute_loss(x, out)
    # The unsqueeze(1) hint only appears when the sample dimension may be missing
    assert ("unsqueeze(1)" in str(exc_info.value)) == suggests_unsqueeze


@pytest.mark.parametrize(
    "mu_shape, log_var_shape",
    [
        ((1, 4), (1, 4)),  # batch 1: the KL was broadcast over the whole batch
        ((3, 4), (3, 3)),  # log_var differs from mu
        ((3, 3), (3, 3)),  # D_z differs from latent_dim
    ],
)
def test_vae_compute_loss_rejects_bad_posterior_shapes(mu_shape, log_var_shape):
    vae = _make_built_vae(in_features=5, latent_dim=4)
    x = torch.randn(3, 5)
    out = VAEOutput(
        x_hat=torch.randn(3, 2, 5),
        z=torch.randn(3, 2, 4),
        mu=torch.randn(mu_shape),
        log_var=torch.randn(log_var_shape),
    )

    with pytest.raises(ValueError, match=r"mu and log_var must have shape \[B, D_z\] = \[3, 4\]"):
        vae.compute_loss(x, out)


def test_adagvae_compute_loss_rejects_bad_shapes():
    from pyautoencoder.variational.vae import AdaGVAE

    adagvae = AdaGVAE(vae=_make_built_vae(in_features=5))
    x1, x2 = torch.randn(3, 5), torch.randn(3, 5)
    out = adagvae((x1, x2))
    out.output2.x_hat = out.output2.x_hat[:, 0]  # drop the sample dimension

    with pytest.raises(ValueError, match=r"x_hat must have shape"):
        adagvae.compute_loss((x1, x2), out)


def test_vae_decode_accepts_single_sample_with_explicit_S_dim():
    vae = _make_built_vae(in_features=5, latent_dim=4)

    dec = vae.decode(torch.randn(3, 4).unsqueeze(1))

    assert dec.x_hat.shape == (3, 1, 5)


def test_vae_rejects_non_flat_encoder_output():
    encoder = nn.Unflatten(1, (2, 3))  # [B, 6] -> [B, 2, 3]
    decoder = DummyDecoder(latent_dim=4, out_features=6)
    vae = VAE(encoder=encoder, decoder=decoder, latent_dim=4)

    with pytest.raises(ValueError, match=r"shape \[B, F\]"):
        vae(torch.randn(4, 6))


def test_adagvae_rejects_invalid_S():
    from pyautoencoder.variational.vae import AdaGVAE

    adagvae = AdaGVAE(vae=_make_built_vae())
    adagvae.train()
    x1, x2 = torch.randn(3, 5), torch.randn(3, 5)

    with pytest.raises(ValueError, match="S must be >= 1"):
        adagvae((x1, x2), S=0)


# ================= AdaGVAE train vs eval =================

def test_adagvae_eval_forward_behaves_as_plain_vae():
    from pyautoencoder.variational.vae import AdaGVAE

    B, S = 4, 3
    adagvae = AdaGVAE(vae=_make_built_vae())
    adagvae.eval()
    x1, x2 = torch.randn(B, 5), torch.randn(B, 5)

    with torch.no_grad():
        torch.manual_seed(0)
        out = adagvae((x1, x2), S=S)
        torch.manual_seed(0)
        ref1 = adagvae.vae(x1, S=S)
        ref2 = adagvae.vae(x2, S=S)

    # Same seed, same results: no grouping, each input through VAE.forward
    for out_i, ref_i in ((out.output1, ref1), (out.output2, ref2)):
        assert torch.equal(out_i.mu, ref_i.mu)
        assert torch.equal(out_i.log_var, ref_i.log_var)
        assert torch.equal(out_i.z, ref_i.z)
        assert torch.equal(out_i.x_hat, ref_i.x_hat)


def test_adagvae_train_forward_shares_lowest_kl_dimension():
    from pyautoencoder.variational.vae import AdaGVAE

    adagvae = AdaGVAE(vae=_make_built_vae(latent_dim=4))
    adagvae.train()
    x1, x2 = torch.randn(6, 5), torch.randn(6, 5)

    out = adagvae((x1, x2), S=2)

    # The dimension with the smallest KL(q1||q2) is always below tau, so it is
    # averaged and identical in both adapted posteriors.
    shared = torch.isclose(out.output1.mu, out.output2.mu)
    assert shared.any(dim=1).all()
    assert not shared.all()


def test_compute_loss_accepts_output_keyword_like_base_class():
    from pyautoencoder.variational.vae import AdaGVAE

    vae = _make_built_vae()
    x1, x2 = torch.randn(3, 5), torch.randn(3, 5)

    assert vae.compute_loss(x1, output=vae(x1)).objective.dim() == 0

    adagvae = AdaGVAE(vae=vae)
    assert adagvae.compute_loss((x1, x2), output=adagvae((x1, x2))).objective.dim() == 0


def test_sampling_layer_class_is_importable_only_from_stochastic_layers():
    # The only import path is pyautoencoder.variational.stochastic_layers
    # (imported at the top of this file); the package and vae.py don't expose it.
    with pytest.raises(ImportError):
        from pyautoencoder.variational import FullyFactorizedGaussian as _  # noqa: F401
    with pytest.raises(ImportError):
        from pyautoencoder.variational.vae import FullyFactorizedGaussian as _  # noqa: F401, F811

    assert isinstance(_make_built_vae().sampling_layer, FullyFactorizedGaussian)
