from dataclasses import dataclass
import pytest
import torch
import torch.nn as nn

from pyautoencoder._base.base import ModelOutput, BaseAutoencoder
from pyautoencoder.loss.base import LossResult

# ================= ModelOutput =================

def test_model_output_repr_tensors_and_non_tensors():
    @dataclass(slots=True, repr=False)
    class MyOutput(ModelOutput):
        logits: torch.Tensor
        labels: torch.Tensor
        meta: dict
        scalar: int | None

    logits = torch.randn(2, 3)
    labels = torch.zeros(2, dtype=torch.long)
    meta = {"a": 1}
    scalar = 42

    out = MyOutput(logits=logits, labels=labels, meta=meta, scalar=scalar)
    s = repr(out)

    assert s.startswith("MyOutput(") and s.endswith(")")

    assert "logits=Tensor(" in s
    assert f"shape={tuple(logits.shape)}" in s
    assert f"dtype={logits.dtype}" in s

    assert "labels=Tensor(" in s
    assert f"shape={tuple(labels.shape)}" in s
    assert f"dtype={labels.dtype}" in s

    assert "meta={'a': 1}" in s
    assert "scalar=42" in s


def test_model_output_repr_empty_dataclass():
    @dataclass(slots=True, repr=False)
    class EmptyOutput(ModelOutput):
        pass

    out = EmptyOutput()
    assert repr(out) == "EmptyOutput()"


def test_model_output_repr_non_tensor_nested():
    @dataclass(slots=True)
    class Nested(ModelOutput):
        nested: list[dict[str, int]]

    obj = Nested(nested=[{"x": 1}, {"y": 2}])
    s = repr(obj)
    assert "Nested(" in s and s.endswith(")")
    assert "nested=[{'x': 1}, {'y': 2}]" in s


# ================= BaseAutoencoder =================

@dataclass(slots=True, repr=False)
class AEEncodeOutput(ModelOutput):
    z: torch.Tensor


@dataclass(slots=True, repr=False)
class AEDecodeOutput(ModelOutput):
    x_hat: torch.Tensor


@dataclass(slots=True, repr=False)
class AEForwardOutput(ModelOutput):
    z: torch.Tensor
    x_hat: torch.Tensor


class ToyAutoencoder(BaseAutoencoder):
    """Minimal concrete BaseAutoencoder implementation for testing."""

    def __init__(self, in_dim: int = 7, latent_dim: int = 14) -> None:
        super().__init__()
        self.encoder = nn.Linear(in_dim, latent_dim, bias=False)
        self.decoder = nn.Linear(latent_dim, in_dim, bias=False)
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.encode_grad_enabled: bool | None = None
        self.decode_grad_enabled: bool | None = None
        self.forward_grad_enabled: bool | None = None

    def _encode(self, x: torch.Tensor) -> AEEncodeOutput:
        self.encode_grad_enabled = torch.is_grad_enabled()
        return AEEncodeOutput(z=self.encoder(x))

    def _decode(self, z: torch.Tensor) -> AEDecodeOutput:
        self.decode_grad_enabled = torch.is_grad_enabled()
        return AEDecodeOutput(x_hat=self.decoder(z))

    def forward(self, x: torch.Tensor) -> AEForwardOutput:
        self.forward_grad_enabled = torch.is_grad_enabled()
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return AEForwardOutput(z=z, x_hat=x_hat)

    def compute_loss(self, x: torch.Tensor, output: AEForwardOutput, *args, **kwargs):
        mse = ((x - output.x_hat) ** 2).mean()
        return LossResult(objective=mse, diagnostics={"mse": mse.item()})


class ToyLazyAutoencoder(BaseAutoencoder):
    """BaseAutoencoder with LazyLinear layers for testing load_state_dict."""

    def __init__(self, latent_dim: int = 8, out_dim: int = 3) -> None:
        super().__init__()
        self.encoder = nn.LazyLinear(latent_dim)
        self.decoder = nn.LazyLinear(out_dim)
        self.latent_dim = latent_dim
        self.out_dim = out_dim

    def _encode(self, x: torch.Tensor) -> AEEncodeOutput:
        return AEEncodeOutput(z=self.encoder(x))

    def _decode(self, z: torch.Tensor) -> AEDecodeOutput:
        return AEDecodeOutput(x_hat=self.decoder(z))

    def forward(self, x: torch.Tensor) -> AEForwardOutput:
        enc = self._encode(x)
        dec = self._decode(enc.z)
        return AEForwardOutput(z=enc.z, x_hat=dec.x_hat)

    def compute_loss(self, x: torch.Tensor, output: AEForwardOutput, *args, **kwargs):
        mse = ((x - output.x_hat) ** 2).mean()
        return LossResult(objective=mse, diagnostics={"mse": mse.item()})


def test_base_autoencoder_is_abstract():
    with pytest.raises(TypeError):
        BaseAutoencoder()  # type: ignore


def test_build_runs_forward_under_no_grad():
    model = ToyAutoencoder()
    x = torch.randn(4, model.in_dim)

    torch.set_grad_enabled(True)
    model.build(x)

    # build() is @torch.no_grad(), so forward sees grad disabled
    assert model.forward_grad_enabled is False
    # global state must be restored
    assert torch.is_grad_enabled() is True


def test_training_paths_use_grad_encode_decode_forward():
    model = ToyAutoencoder()
    x = torch.randn(5, model.in_dim)

    torch.set_grad_enabled(True)

    out_f = model.forward(x)
    assert isinstance(out_f, AEForwardOutput)
    assert out_f.z.shape == (5, model.latent_dim)
    assert out_f.x_hat.shape == x.shape
    assert model.forward_grad_enabled is True

    out_e = model._encode(x)
    assert isinstance(out_e, AEEncodeOutput)
    assert out_e.z.shape == (5, model.latent_dim)
    assert model.encode_grad_enabled is True
    assert out_e.z.requires_grad is True

    out_d = model._decode(out_e.z)
    assert isinstance(out_d, AEDecodeOutput)
    assert out_d.x_hat.shape == x.shape
    assert model.decode_grad_enabled is True
    assert out_d.x_hat.requires_grad is True


def test_inference_encode_decode_use_inference_mode_no_grad():
    model = ToyAutoencoder()
    x = torch.randn(5, model.in_dim)

    torch.set_grad_enabled(True)
    assert torch.is_grad_enabled() is True

    encode_out = model.encode(x)
    assert isinstance(encode_out, AEEncodeOutput)
    assert encode_out.z.shape == (5, model.latent_dim)

    decode_out = model.decode(encode_out.z)
    assert isinstance(decode_out, AEDecodeOutput)
    assert decode_out.x_hat.shape == x.shape

    # _encode/_decode bodies must see grad disabled
    assert model.encode_grad_enabled is False
    assert model.decode_grad_enabled is False

    # Outputs from inference_mode must not require grad
    assert encode_out.z.requires_grad is False
    assert decode_out.x_hat.requires_grad is False

    # Global grad state must be restored
    assert torch.is_grad_enabled() is True


def test_load_state_dict_raises_for_uninitialized_lazy_modules():
    model = ToyLazyAutoencoder()

    with pytest.raises(RuntimeError, match="Call build\\(\\) before load_state_dict"):
        model.load_state_dict({})


def test_load_state_dict_works_after_build():
    model1 = ToyAutoencoder()
    x = torch.randn(4, model1.in_dim)
    model1.build(x)
    state = model1.state_dict()

    model2 = ToyAutoencoder()
    model2.build(x)

    result = model2.load_state_dict(state, strict=True, assign=False)
    assert result is not None
