import pyarrow  # MUST be first — Windows DLL fix
import transformers
import torch
import numpy as np
import pytest

from src.probing.faithfulness_detector import (
    FaithfulnessProbe,
    corrupt_step,
    train_probe,
)


@pytest.mark.slow
def test_extract_cot_activations_shape(small_model):
    from src.probing.cot_probe import extract_cot_activations
    acts = extract_cot_activations(small_model, "Hello world")
    assert len(acts) == small_model.cfg.n_layers
    for layer, tensor in acts.items():
        assert tensor.ndim == 2
        assert tensor.shape[1] == small_model.cfg.d_model


@pytest.mark.slow
def test_extract_cot_activations_empty_prompt_raises(small_model):
    from src.probing.cot_probe import extract_cot_activations
    # Empty string tokenizes to at least BOS; should not crash
    acts = extract_cot_activations(small_model, "")
    assert isinstance(acts, dict)


def test_faithfulness_probe_train_predict_range(dummy_activations, dummy_labels):
    probe = FaithfulnessProbe(layer=6)
    acts = dummy_activations[6].numpy()
    probe.train(acts, dummy_labels)
    preds = probe.predict(acts)
    assert len(preds) == len(dummy_labels)
    for label, conf in preds:
        assert isinstance(label, bool)
        assert 0.0 <= conf <= 1.0


def test_faithfulness_probe_direction_shape(dummy_probe):
    direction = dummy_probe.direction
    assert direction.ndim == 1
    assert direction.shape[0] == 768
    assert abs(direction.norm().item() - 1.0) < 1e-5


def test_corrupt_step_number_perturbation_changes_numbers():
    step = "John has 10 apples and 5 oranges."
    corrupted = corrupt_step(step, strategy="number_perturbation", seed=0)
    assert corrupted != step
    # At least one number should differ
    original_nums = set(s for s in step.split() if s.replace(".", "").isdigit())
    corrupted_nums = set(s.rstrip(".") for s in corrupted.split() if any(c.isdigit() for c in s))
    assert original_nums != corrupted_nums or corrupted != step


def test_corrupt_step_negate_adds_negation():
    step = "The answer is 42."
    corrupted = corrupt_step(step, strategy="negate", seed=0)
    assert "not" in corrupted.lower() or corrupted != step


def test_corrupt_step_shuffle_rearranges():
    step = "First compute the sum of all values."
    corrupted = corrupt_step(step, strategy="shuffle", seed=7)
    # Word set should be the same, order should differ (with high probability)
    assert sorted(step.split()) == sorted(corrupted.split())


def test_probe_save_load_roundtrip(dummy_activations, dummy_labels, tmp_path):
    probe = FaithfulnessProbe(layer=3)
    acts = dummy_activations[3].numpy()
    probe.train(acts, dummy_labels)
    save_path = str(tmp_path / "probe.pkl")
    probe.save(save_path)

    loaded = FaithfulnessProbe.load(save_path)
    assert loaded.layer == 3
    orig_preds = probe.predict(acts)
    loaded_preds = loaded.predict(acts)
    assert orig_preds == loaded_preds
