import pyarrow  # MUST be first — Windows DLL fix
import transformers
import torch
import pytest

from src.steering.steering_vectors import (
    compute_steering_vector,
    compute_steering_vector_from_dicts,
    save_steering_vector,
    load_steering_vector,
)


@pytest.fixture
def faithful_acts():
    torch.manual_seed(1)
    return torch.randn(8, 768) + 2.0


@pytest.fixture
def unfaithful_acts():
    torch.manual_seed(2)
    return torch.randn(8, 768) - 2.0


def test_compute_steering_vector_shape(faithful_acts, unfaithful_acts):
    v = compute_steering_vector(faithful_acts, unfaithful_acts, layer=6)
    assert v.shape == (768,)


def test_compute_steering_vector_normalized(faithful_acts, unfaithful_acts):
    v = compute_steering_vector(faithful_acts, unfaithful_acts, layer=6)
    assert abs(v.norm().item() - 1.0) < 1e-5


def test_compute_steering_vector_from_dicts_shape(dummy_activations):
    # Use offset copies so faithful_mean != unfaithful_mean (avoids zero-norm direction)
    faithful_dicts = [{l: acts + 1.0 for l, acts in dummy_activations.items()} for _ in range(4)]
    unfaithful_dicts = [{l: acts - 1.0 for l, acts in dummy_activations.items()} for _ in range(4)]
    v = compute_steering_vector_from_dicts(faithful_dicts, unfaithful_dicts, layer=6)
    assert v.shape == (768,)
    assert abs(v.norm().item() - 1.0) < 1e-5


def test_compute_steering_vector_from_dicts_shape_direct():
    torch.manual_seed(3)
    faithful_dicts = [{l: torch.randn(10, 768) + 1.0 for l in range(12)} for _ in range(5)]
    unfaithful_dicts = [{l: torch.randn(10, 768) - 1.0 for l in range(12)} for _ in range(5)]
    v = compute_steering_vector_from_dicts(faithful_dicts, unfaithful_dicts, layer=6)
    assert v.shape == (768,)
    assert abs(v.norm().item() - 1.0) < 1e-5


def test_steering_vector_save_load_roundtrip(tmp_path, faithful_acts, unfaithful_acts):
    v = compute_steering_vector(faithful_acts, unfaithful_acts, layer=6)
    meta = {"layer": 6, "model": "gpt2"}
    path = str(tmp_path / "vec.pt")
    save_steering_vector(v, path, metadata=meta)

    loaded_v, loaded_meta = load_steering_vector(path)
    assert torch.allclose(v, loaded_v)
    assert loaded_meta["layer"] == 6


@pytest.mark.slow
def test_apply_steering_changes_logits(small_model):
    from src.steering.activation_steering import steered_forward
    tokens = small_model.to_tokens("Hello world")
    torch.manual_seed(42)
    v = torch.randn(small_model.cfg.d_model)
    v = v / v.norm()

    with torch.no_grad():
        baseline_logits = small_model(tokens).clone()
    steered_logits = steered_forward(small_model, tokens, v, layer=6, alpha=5.0)

    assert not torch.allclose(baseline_logits, steered_logits)


@pytest.mark.slow
def test_apply_steering_alpha_zero_unchanged(small_model):
    from src.steering.activation_steering import steered_forward
    tokens = small_model.to_tokens("Hello world")
    torch.manual_seed(42)
    v = torch.randn(small_model.cfg.d_model)

    with torch.no_grad():
        baseline_logits = small_model(tokens).clone()
    steered_logits = steered_forward(small_model, tokens, v, layer=6, alpha=0.0)

    assert torch.allclose(baseline_logits, steered_logits, atol=1e-5)


@pytest.mark.slow
def test_apply_steering_hooks_cleaned_up(small_model):
    """Verify reset_hooks() removes the injected hook after the context exits."""
    from src.steering.activation_steering import apply_steering
    tokens = small_model.to_tokens("Hook cleanup test")
    torch.manual_seed(42)
    v = torch.randn(small_model.cfg.d_model)

    # Establish a clean baseline
    with torch.no_grad():
        logits_before = small_model(tokens).clone()

    # Run with a large alpha so steering definitely changes logits
    with apply_steering(small_model, v, layer=6, alpha=20.0):
        with torch.no_grad():
            logits_during = small_model(tokens).clone()

    # After context exit, forward pass must be identical to pre-steering baseline
    with torch.no_grad():
        logits_after = small_model(tokens).clone()

    assert not torch.allclose(logits_before, logits_during, atol=1e-4), \
        "Steering should have altered logits while the hook was active"
    assert torch.allclose(logits_before, logits_after, atol=1e-5), \
        "Logits after context must match baseline — hook not cleaned up"
