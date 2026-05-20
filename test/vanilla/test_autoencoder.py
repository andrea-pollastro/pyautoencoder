import pytest
import torch
import torch.nn as nn

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

def test_ae_build_does_not_replace_modules():
    encoder = SimpleEncoder(in_features=4, latent_features=6)
    decoder = SimpleDecoder(latent_features=6, out_features=4)
    ae = AE(encoder=encoder, decoder=decoder)

    encoder_id = id(ae.encoder)
    decoder_id = id(ae.decoder)

    ae.build(torch.randn(1, 4))

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

    torch.set_grad_enabled(True)

    enc_out = ae._encode(x)
    assert isinstance(enc_out, AEEncodeOutput)
    assert enc_out.z.shape == (batch_size, latent_features)
    assert enc_out.z.requires_grad is True

    dec_out = ae._decode(enc_out.z)
    assert isinstance(dec_out, AEDecodeOutput)
    assert dec_out.x_hat.shape == (batch_size, in_features)
    assert dec_out.x_hat.requires_grad is True

    out = ae.forward(x)
    assert isinstance(out, AEOutput)
    assert out.z.shape == (batch_size, latent_features)
    assert out.x_hat.shape == (batch_size, in_features)
    assert out.z.requires_grad is True
    assert out.x_hat.requires_grad is True

    assert encoder.last_grad_enabled is True
    assert decoder.last_grad_enabled is True


def test_ae_forward_calls_encoder_then_decoder_consistently():
    batch_size = 4
    in_features = 7
    latent_features = 5

    encoder = SimpleEncoder(in_features=in_features, latent_features=latent_features)
    decoder = SimpleDecoder(latent_features=latent_features, out_features=in_features)
    ae = AE(encoder=encoder, decoder=decoder)

    x = torch.randn(batch_size, in_features)

    torch.set_grad_enabled(True)

    enc_out = ae._encode(x)
    dec_out = ae._decode(enc_out.z)
    out = ae.forward(x)

    assert torch.allclose(out.z, enc_out.z)
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

    assert encoder.last_grad_enabled is False
    assert decoder.last_grad_enabled is False

    assert torch.is_grad_enabled() is True


def test_ae_output_repr_uses_modeloutput_smart_repr():
    batch_size = 2
    in_features = 3
    latent_features = 4

    x_hat = torch.randn(batch_size, in_features)
    z = torch.randn(batch_size, latent_features)

    out = AEOutput(x_hat=x_hat, z=z)
    s = repr(out)

    assert s.startswith("AEOutput(") and s.endswith(")")
    assert "x_hat=Tensor(" in s
    assert f"shape={tuple(x_hat.shape)}" in s
    assert f"dtype={x_hat.dtype}" in s

    assert "z=Tensor(" in s
    assert f"shape={tuple(z.shape)}" in s
    assert f"dtype={z.dtype}" in s


# ================= compute_loss =================

def test_ae_compute_loss_gaussian_likelihood_returns_correct_type():
    batch_size = 4
    in_features = 8
    latent_features = 3

    encoder = SimpleEncoder(in_features=in_features, latent_features=latent_features)
    decoder = SimpleDecoder(latent_features=latent_features, out_features=in_features)
    ae = AE(encoder=encoder, decoder=decoder)

    x = torch.randn(batch_size, in_features)

    torch.set_grad_enabled(True)
    ae_output = ae.forward(x)

    loss_result = ae.compute_loss(x, ae_output)

    from pyautoencoder.loss.base import LossResult
    assert isinstance(loss_result, LossResult)
    assert hasattr(loss_result, 'objective')
    assert hasattr(loss_result, 'diagnostics')

    assert loss_result.objective.dim() == 0
    assert loss_result.objective.requires_grad is True

    assert isinstance(loss_result.diagnostics, dict)
    assert 'log_likelihood' in loss_result.diagnostics
    assert isinstance(loss_result.diagnostics['log_likelihood'], float)


def test_ae_compute_loss_gaussian_nll_equals_half_mse():
    batch_size = 5
    in_features = 6
    latent_features = 2

    encoder = SimpleEncoder(in_features=in_features, latent_features=latent_features)
    decoder = SimpleDecoder(latent_features=latent_features, out_features=in_features)
    ae = AE(encoder=encoder, decoder=decoder)

    x = torch.randn(batch_size, in_features)

    torch.set_grad_enabled(True)
    ae_output = ae.forward(x)

    loss_result = ae.compute_loss(x, ae_output, likelihood='gaussian')

    expected = 0.5 * ((ae_output.x_hat - x) ** 2).reshape(batch_size, -1).sum(-1).mean()
    assert torch.allclose(loss_result.objective, expected, atol=1e-6)


def test_ae_compute_loss_bernoulli_nll_equals_bce():
    import torch.nn.functional as F

    batch_size = 4
    in_features = 8
    latent_features = 3

    encoder = SimpleEncoder(in_features=in_features, latent_features=latent_features)
    decoder = SimpleDecoder(latent_features=latent_features, out_features=in_features)
    ae = AE(encoder=encoder, decoder=decoder)

    x = torch.sigmoid(torch.randn(batch_size, in_features))
    ae_output = ae.forward(x)

    loss_result = ae.compute_loss(x, ae_output, likelihood='bernoulli')

    expected = F.binary_cross_entropy_with_logits(
        ae_output.x_hat, x, reduction='none'
    ).reshape(batch_size, -1).sum(-1).mean()

    assert loss_result.objective.dim() == 0
    assert torch.allclose(loss_result.objective, expected, atol=1e-6)
    assert 'log_likelihood' in loss_result.diagnostics


def test_ae_compute_loss_backward_flows_through_all_params():
    batch_size = 2
    in_features = 4
    latent_features = 2

    encoder = SimpleEncoder(in_features=in_features, latent_features=latent_features)
    decoder = SimpleDecoder(latent_features=latent_features, out_features=in_features)
    ae = AE(encoder=encoder, decoder=decoder)

    x = torch.randn(batch_size, in_features)

    torch.set_grad_enabled(True)
    ae_output = ae.forward(x)

    loss_result = ae.compute_loss(x, ae_output)
    loss_result.objective.backward()

    enc_grads = [p.grad for p in encoder.parameters() if p.requires_grad]
    dec_grads = [p.grad for p in decoder.parameters() if p.requires_grad]
    assert any(g is not None and torch.any(g != 0) for g in enc_grads)
    assert any(g is not None and torch.any(g != 0) for g in dec_grads)


def test_ae_compute_loss_diagnostics_log_likelihood_negative_of_objective():
    batch_size = 3
    in_features = 5
    latent_features = 2

    encoder = SimpleEncoder(in_features=in_features, latent_features=latent_features)
    decoder = SimpleDecoder(latent_features=latent_features, out_features=in_features)
    ae = AE(encoder=encoder, decoder=decoder)

    x = torch.randn(batch_size, in_features)

    torch.set_grad_enabled(True)
    ae_output = ae.forward(x)

    loss_result = ae.compute_loss(x, ae_output)

    nll = loss_result.objective.item()
    ll = loss_result.diagnostics['log_likelihood']
    assert torch.allclose(torch.tensor(ll), torch.tensor(-nll), atol=1e-6)


def test_ae_compute_loss_batch_size_one():
    in_features = 4
    latent_features = 2

    encoder = SimpleEncoder(in_features=in_features, latent_features=latent_features)
    decoder = SimpleDecoder(latent_features=latent_features, out_features=in_features)
    ae = AE(encoder=encoder, decoder=decoder)

    x = torch.randn(1, in_features)

    torch.set_grad_enabled(True)
    ae_output = ae.forward(x)

    loss_result = ae.compute_loss(x, ae_output)

    assert loss_result.objective.dim() == 0
    assert not torch.isnan(loss_result.objective)
    assert not torch.isinf(loss_result.objective)
