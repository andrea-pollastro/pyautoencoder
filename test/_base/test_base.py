from typing import Any, Mapping
from dataclasses import dataclass
import pytest
import torch
import torch.nn as nn

from pyautoencoder._base.base import (
    ModelOutput, 
    BuildGuardMixin, 
    NotBuiltError,
    BaseAutoencoder
)

# ================= ModelOutput =================

def test_model_output_repr_tensors_and_non_tensors():
    @torch.no_grad()
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


# ================= BuildGuardMixin =================

def test_guarded_method_raises_before_build_and_works_after_build():
    class MyModel(BuildGuardMixin):
        _GUARDED = ("forward", "extra")

        def __init__(self):
            super().__init__()
            self.was_built_with: Any = None
            self.built_under_no_grad: bool | None = None

        def build(self, x):
            # We check that this is executed under no_grad
            self.was_built_with = x
            self.built_under_no_grad = not torch.is_grad_enabled()
            self._built = True

        def forward(self, x):
            return x * 2

        def extra(self):
            return "ok"

    m = MyModel()

    # before building
    assert m._built is False
    with pytest.raises(NotBuiltError):
        m.forward(torch.tensor([1.0]))
    with pytest.raises(NotBuiltError):
        m.extra()

    # after building
    x = torch.tensor([1.0, 2.0])
    m.build(x)
    assert m._built is True
    assert torch.equal(m.was_built_with, x)
    assert m.built_under_no_grad is True

    out = m.forward(x)
    assert torch.equal(out, x * 2)
    assert m.extra() == "ok"


def test_guarded_method_swaps_to_original_after_first_successful_call():
    class SwapModel(BuildGuardMixin):
        _GUARDED = ("forward",)

        def __init__(self):
            super().__init__()
            self.build_calls = 0
            self.forward_calls = 0

        def build(self, x):
            self.build_calls += 1
            self._built = True

        def forward(self, x):
            self.forward_calls += 1
            return x + 1

    m = SwapModel()
    x = torch.tensor([1.0])

    # building
    m.build(x)
    assert m.build_calls == 1

    out1 = m.forward(x)
    assert torch.equal(out1, x + 1)
    assert m.forward_calls == 1

    # After the first successful call, the instance should have a direct bound method
    # (i.e., the guard is no longer invoked).
    # To test that, toggling `_built` back to False should not break subsequent calls.
    m._built = False  # would fail if guard were still in place
    out2 = m.forward(x)
    assert torch.equal(out2, x + 1)
    assert m.forward_calls == 2


def test_build_raises_if_subclass_does_not_set_built_flag():
    class BadBuild(BuildGuardMixin):
        _GUARDED = ("forward",)

        def __init__(self):
            super().__init__()

        def build(self, x):
            self.x = x
            # Forgot to set self._built = True

        def forward(self, x):
            return x

    m = BadBuild()

    # Calling build should raise NotBuiltError because _built is still False
    with pytest.raises(NotBuiltError):
        m.build(torch.tensor([1.0]))

    # _built remains False
    assert m._built is False


def test_build_is_no_op_if_already_built():
    class IdempotentModel(BuildGuardMixin):
        _GUARDED = ("forward",)

        def __init__(self):
            super().__init__()
            self.build_calls = 0

        def build(self, x):
            self.build_calls += 1
            self._built = True

        def forward(self, x):
            return x * 3

    m = IdempotentModel()

    # First build
    m.build(torch.tensor([1.0]))
    assert m.build_calls == 1
    assert m._built is True

    # Second build: wrapper sees _built=True and returns early without calling _orig_build
    m.build(torch.tensor([2.0]))
    assert m.build_calls == 1  # still 1


def test_missing_guarded_class_attribute_raises_type_error():
    # __init_subclass__ runs at class creation time, so we have to define it inside the with block
    with pytest.raises(TypeError, match="must define a class attribute `_GUARDED`"):
        class NoGuard(BuildGuardMixin):
            def forward(self, x):
                return x


def test_non_iterable_guarded_raises_type_error():
    with pytest.raises(TypeError, match="must be an iterable of method names"):
        class NonIterableGuarded(BuildGuardMixin):
            _GUARDED = 123  # not iterable

            def forward(self, x):
                return x

    # Strings and bytes are explicitly disallowed (they are iterables of chars but semantically wrong)
    with pytest.raises(TypeError, match="must be an iterable of method names"):
        class StringGuarded(BuildGuardMixin):
            _GUARDED = "forward"  # string is not allowed

            def forward(self, x):
                return x

    with pytest.raises(TypeError, match="must be an iterable of method names"):
        class BytesGuarded(BuildGuardMixin):
            _GUARDED = b"forward"  # bytes is not allowed

            def forward(self, x):
                return x

def test_guarded_is_coerced_to_set_and_de_duplicated():
    class DupGuard(BuildGuardMixin):
        _GUARDED = ["forward", "forward", "extra"]

        def __init__(self):
            super().__init__()

        def build(self, x):
            self._built = True

        def forward(self, x):
            return x

        def extra(self):
            return "ok"

    # After class creation, _GUARDED should be a set
    assert isinstance(DupGuard._GUARDED, set)
    assert DupGuard._GUARDED == {"forward", "extra"}

def test_inheritance_keeps_guard_behavior_for_inherited_methods():
    class ParentModel(BuildGuardMixin):
        _GUARDED = ("forward",)

        def __init__(self):
            super().__init__()

        def build(self, x):
            self._built = True

        def forward(self, x):
            return x + 10

    class ChildModel(ParentModel):
        # Inherits _GUARDED and methods; no overrides
        pass

    c = ChildModel()

    # Guard still applies on subclass instances
    with pytest.raises(NotBuiltError):
        c.forward(torch.tensor([1.0]))

    c.build(torch.tensor([1.0]))
    out = c.forward(torch.tensor([2.0]))
    assert torch.equal(out, torch.tensor([12.0]))


def test_child_can_extend_guarded_set_and_add_new_guarded_method():
    class Parent(BuildGuardMixin):
        _GUARDED = {"forward"}

        def __init__(self):
            super().__init__()

        def build(self, x):
            self._built = True

        def forward(self, x):
            return x * 2

    class Child(Parent):
        _GUARDED = Parent._GUARDED.union({"extra"})  # extend guarded methods

        def extra(self, x):
            return x * 3

    # After class creation, Child._GUARDED should be a set containing both
    assert Child._GUARDED == {"forward", "extra"}

    m = Child()

    # Both methods should be guarded
    with pytest.raises(NotBuiltError):
        m.forward(torch.tensor([1.0]))
    with pytest.raises(NotBuiltError):
        m.extra(torch.tensor([1.0]))

    # After build, both should work
    m.build(torch.tensor([1.0]))
    assert torch.equal(m.forward(torch.tensor([2.0])), torch.tensor([4.0]))
    assert torch.equal(m.extra(torch.tensor([2.0])), torch.tensor([6.0]))

def test_guarded_method_preserves_function_metadata():
    class M(BuildGuardMixin):
        _GUARDED = ("forward",)
        def build(self,x): self._built=True
        def forward(self,x): return x

    m = M()
    assert m.forward.__name__ == "forward"

def test_build_accepts_args_kwargs():
    class M(BuildGuardMixin):
        _GUARDED=("forward",)
        def forward(self,x): return x
        def build(self, x, scale=1): 
            self.scale = scale
            self._built=True

    m = M()
    m.build(torch.tensor([1]), scale=3)
    assert m.scale == 3

def test_guarded_missing_methods_are_safely_ignored():
    class M(BuildGuardMixin):
        _GUARDED = ("forward", "imaginary_method")
        def build(self,x): self._built=True
        def forward(self,x): return x

    m = M()
    m.build(1)
    assert m.forward(2) == 2  # works
    assert not hasattr(m, "imaginary_method")


def test_guard_method_returning_none_still_replaces_itself():
    class M(BuildGuardMixin):
        _GUARDED=("f",)
        def build(self,x): self._built=True
        def f(self): return None

    m=M()
    m.build(None)
    assert m.f() is None   # should not raise, and should replace guard
    m._built=False
    assert m.f() is None   # guard removed

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
    """
    Minimal concrete BaseAutoencoder implementation to test class' behavior
    """

    def __init__(self) -> None:
        super().__init__()
        # created in build()
        self.encoder: nn.Linear | None = None
        self.decoder: nn.Linear | None = None

        # for introspection in tests
        self.build_called: bool = False
        self.build_grad_enabled: bool | None = None
        self.encode_grad_enabled: bool | None = None
        self.decode_grad_enabled: bool | None = None
        self.forward_grad_enabled: bool | None = None

    def build(self, input_sample: torch.Tensor) -> None:
        # should run under torch.no_grad() due to BuildGuardMixin wrapper
        in_dim = input_sample.shape[-1]
        self.encoder = nn.Linear(in_dim, in_dim * 2, bias=False)
        self.decoder = nn.Linear(in_dim * 2, in_dim, bias=False)

        self.build_called = True
        self.build_grad_enabled = torch.is_grad_enabled()
        self._built = True

    def _encode(self, x: torch.Tensor) -> ModelOutput:
        assert self.encoder is not None
        self.encode_grad_enabled = torch.is_grad_enabled()
        z = self.encoder(x)
        return AEEncodeOutput(z=z)

    def _decode(self, z: torch.Tensor) -> ModelOutput:
        assert self.decoder is not None
        self.decode_grad_enabled = torch.is_grad_enabled()
        x_hat = self.decoder(z)
        return AEDecodeOutput(x_hat=x_hat)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        assert self.encoder is not None and self.decoder is not None
        self.forward_grad_enabled = torch.is_grad_enabled()
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return AEForwardOutput(z=z, x_hat=x_hat)

    def compute_loss(self, x: torch.Tensor, model_output: ModelOutput, *args, **kwargs):
        """Dummy loss function for testing."""
        # Simple dummy loss: mean squared error between input and reconstruction
        x_hat = model_output.x_hat  # type: ignore
        mse = ((x - x_hat) ** 2).mean()
        from pyautoencoder.loss.base import LossResult
        return LossResult(
            objective=mse,
            diagnostics={"mse": mse.item()}
        )


def test_base_autoencoder_is_abstract():
    # Cannot instantiate BaseAutoencoder directly due to abstract methods
    with pytest.raises(TypeError):
        BaseAutoencoder()  # type: ignore


def test_base_autoencoder_guarded_set():
    # BaseAutoencoder declares the guarded methods
    assert BaseAutoencoder._GUARDED == {"forward", "_encode", "_decode"}


def test_guarded_methods_raise_before_build():
    model = ToyAutoencoder()
    x = torch.randn(4, 3)

    # BuildGuardMixin should wrap forward/_encode/_decode
    with pytest.raises(NotBuiltError):
        model.forward(x)

    with pytest.raises(NotBuiltError):
        model._encode(x)

    with pytest.raises(NotBuiltError):
        model._decode(x)

    # encode/decode use the guarded _encode/_decode under the hood
    with pytest.raises(NotBuiltError):
        model.encode(x)

    with pytest.raises(NotBuiltError):
        model.decode(x)

def test_build_sets_built_flag_and_runs_under_no_grad():
    model = ToyAutoencoder()
    x = torch.randn(4, 3)

    # Initially not built
    assert model._built is False

    # Global grad enabled, but build() wrapper should use no_grad
    torch.set_grad_enabled(True)
    model.build(x)

    assert model.build_called is True
    # build() body should see grad disabled
    assert model.build_grad_enabled is False

    assert model._built is True
    assert model.encoder is not None
    assert model.decoder is not None


def test_build_idempotent_if_already_built():
    model = ToyAutoencoder()
    x = torch.randn(4, 3)

    # First build
    model.build(x)
    encoder_id = id(model.encoder)
    decoder_id = id(model.decoder)

    # Second build should be a no-op due to BuildGuardMixin wrapper
    # (it returns early if _built is already True)
    model.build(x)
    assert id(model.encoder) == encoder_id
    assert id(model.decoder) == decoder_id


def test_training_paths_use_grad_encode_decode_forward():
    model = ToyAutoencoder()
    x = torch.randn(5, 7)

    model.build(x)

    torch.set_grad_enabled(True)

    # forward
    out_f = model.forward(x)
    assert isinstance(out_f, AEForwardOutput)
    assert out_f.z.shape == (5, 14)
    assert out_f.x_hat.shape == x.shape
    assert model.forward_grad_enabled is True

    # _encode
    out_e = model._encode(x)
    assert isinstance(out_e, AEEncodeOutput)
    assert out_e.z.shape == (5, 14)
    assert model.encode_grad_enabled is True
    assert out_e.z.requires_grad is True

    # _decode
    out_d = model._decode(out_e.z)
    assert isinstance(out_d, AEDecodeOutput)
    assert out_d.x_hat.shape == x.shape
    assert model.decode_grad_enabled is True
    assert out_d.x_hat.requires_grad is True


def test_inference_encode_decode_use_inference_mode_no_grad():
    model = ToyAutoencoder()
    x = torch.randn(5, 7)

    model.build(x)

    # Global grad enabled, but encode/decode are @torch.inference_mode()
    torch.set_grad_enabled(True)
    assert torch.is_grad_enabled() is True

    # encode
    encode_out = model.encode(x)
    assert isinstance(encode_out, AEEncodeOutput)
    assert encode_out.z.shape == (5, 14)

    # decode
    decode_out = model.decode(encode_out.z)
    assert isinstance(decode_out, AEDecodeOutput)
    assert decode_out.x_hat.shape == x.shape

    # _encode/_decode bodies should see grad disabled
    assert model.encode_grad_enabled is False
    assert model.decode_grad_enabled is False

    # Outputs from inference mode should not require grad
    assert encode_out.z.requires_grad is False
    assert decode_out.x_hat.requires_grad is False

    # Global grad state should be restored
    assert torch.is_grad_enabled() is True


def test_encode_decode_call_guarded_implementation_not_bypassing_guard():
    model = ToyAutoencoder()
    x = torch.randn(3, 4)

    # Before build: encode/decode must still fail because _encode/_decode are guarded
    with pytest.raises(NotBuiltError):
        model.encode(x)

    with pytest.raises(NotBuiltError):
        model.decode(x)

    # After build: they should work
    model.build(x)

    encode_out = model.encode(x)
    decode_out = model.decode(encode_out.z) # type: ignore

    assert isinstance(encode_out, AEEncodeOutput)
    assert isinstance(decode_out, AEDecodeOutput)


def test_built_property_reflects_internal_flag():
    model = ToyAutoencoder()
    assert model._built is False

    model.build(torch.randn(2, 3))
    assert model._built is True


def test_load_state_dict_raises_if_not_built():
    model = ToyAutoencoder()

    with pytest.raises(
        NotBuiltError,
        match="load_state_dict called before build\\(\\)",
    ):
        model.load_state_dict({})


def test_load_state_dict_works_after_build_with_real_state_dict():
    model1 = ToyAutoencoder()
    x = torch.randn(4, 3)
    model1.build(x)
    state = model1.state_dict()

    # Build another with the same input shape and load
    model2 = ToyAutoencoder()
    model2.build(x)

    # Should not raise
    result = model2.load_state_dict(state, strict=True, assign=False)

    assert result is not None

