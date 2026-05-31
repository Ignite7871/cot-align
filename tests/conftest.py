import pyarrow  # MUST be first — Windows DLL fix
import transformers
import torch
import numpy as np
import pytest

from src.probing.faithfulness_detector import FaithfulnessProbe


@pytest.fixture(scope="session")
def small_model():
    """
    Load GPT-2 once per test session via TransformerLens.

    Session-scoped so the model is loaded only once regardless of how many
    tests use it. Tests that use this fixture should be decorated with
    @pytest.mark.slow and deselected with -m "not slow" for fast CI runs.
    """
    from src.utils.model_loader import get_model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return get_model("gpt2", device=device)


@pytest.fixture
def dummy_activations() -> dict:
    """
    Synthetic activation dict matching GPT-2's architecture (12 layers, d=768).

    Shape: {layer_idx: Tensor(10, 768)} — 10 sequence positions, 768-dim residual stream.
    """
    torch.manual_seed(0)
    return {layer: torch.randn(10, 768) for layer in range(12)}


@pytest.fixture
def dummy_labels() -> list[bool]:
    """Alternating faithful/unfaithful labels for 10 samples."""
    return [True, False, True, False, True, False, True, False, True, False]


@pytest.fixture
def dummy_probe(dummy_activations: dict, dummy_labels: list[bool]) -> FaithfulnessProbe:
    """
    FaithfulnessProbe trained on synthetic GPT-2-shaped activations at layer 6.

    Uses layer 6 activations from dummy_activations (shape 10×768) with
    dummy_labels. The probe's accuracy on random data will be near chance —
    this fixture is for testing API correctness, not performance.
    """
    probe = FaithfulnessProbe(layer=6)
    acts = dummy_activations[6].numpy()
    probe.train(acts, dummy_labels)
    return probe
